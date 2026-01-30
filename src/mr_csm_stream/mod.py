"""
mr_csm_stream - MindRoot TTS plugin using CSM with audio context.

Provides streaming TTS using CSM (Conversational Speech Model) which
incorporates user audio context for more natural conversational responses.

Key feature: Streams user audio to CSM server continuously (parallel to
Deepgram STT) so there's zero upload latency when generation starts.
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, Any, Optional, AsyncGenerator

from lib.providers.services import service, service_manager
from lib.providers.commands import command
from lib.providers.hooks import hook
from lib.pipelines.pipe import pipe

from .csm_client import CSMClient
from .audio_pacer import AudioPacer

logger = logging.getLogger(__name__)

# Global clients per session
_csm_clients: Dict[str, CSMClient] = {}
_active_speak_locks: Dict[str, asyncio.Lock] = {}

# Configuration from environment
CSM_SERVER_URL = os.environ.get('MR_CSM_SERVER_URL', 'ws://localhost:8765/ws')
CSM_REF_AUDIO = os.environ.get('MR_CSM_REF_AUDIO', '')
# Set to '1' to bypass AudioPacer and send directly to SIP (for debugging latency)
CSM_BYPASS_PACER = os.environ.get('MR_CSM_BYPASS_PACER', '1').lower() in ('1', 'true', 'yes')


CSM_REF_TEXT = os.environ.get('MR_CSM_REF_TEXT', '')
CSM_SPEAKER_ID = int(os.environ.get('MR_CSM_SPEAKER_ID', '0'))


# Cache for transcribed reference audio
_ref_text_cache: Dict[str, str] = {}

async def get_voice_path_from_context(context) -> Optional[str]:
    """Get the voice audio path from the agent's persona.
    
    The persona's voice_id field should contain an absolute path to the
    reference audio file.
    """
    if not context or not hasattr(context, 'agent_name'):
        return None
    
    try:
        agent_data = await service_manager.get_agent_data(context.agent_name)
        persona = agent_data.get("persona", {})
        
        # voice_id in persona should be a path to the audio file
        voice_path = persona.get("voice_id", "")
        
        if voice_path and os.path.isabs(voice_path) and os.path.exists(voice_path):
            logger.debug(f"Got voice path from persona: {voice_path}")
            return voice_path
        elif voice_path:
            logger.warning(f"Voice path from persona not found or not absolute: {voice_path}")
        
    except Exception as e:
        logger.warning(f"Could not get voice path from persona: {e}")
    
    return None


async def get_ref_text_from_context(context) -> Optional[str]:
    """Get the reference text from the agent's persona.
    
    The persona's voice_ref_text field should contain the transcript of the
    reference audio.
    """
    if not context or not hasattr(context, 'agent_name'):
        return None
    
    try:
        agent_data = await service_manager.get_agent_data(context.agent_name)
        persona = agent_data.get("persona", {})
        
        ref_text = persona.get("voice_ref_text", "")
        
        if ref_text:
            logger.debug(f"Got ref text from persona: {ref_text[:50]}...")
            return ref_text
        
    except Exception as e:
        logger.warning(f"Could not get ref text from persona: {e}")
    
    return None


def _load_ref_audio(audio_path: str) -> Optional[bytes]:
    """Load reference audio file."""
    with open(audio_path, 'rb') as f:
        return f.read()


async def transcribe_reference_audio(audio_path: str) -> Optional[str]:
    """Transcribe reference audio file using Deepgram.
    
    Results are cached by file path to avoid re-transcribing.
    """
    global _ref_text_cache
    
    if audio_path in _ref_text_cache:
        logger.info(f"Using cached transcription for {audio_path}")
        return _ref_text_cache[audio_path]
    
    if not audio_path or not os.path.exists(audio_path):
        return None
    
    try:
        from deepgram import DeepgramClient, PrerecordedOptions, FileSource
        
        api_key = os.environ.get('DEEPGRAM_API_KEY')
        if not api_key:
            logger.warning("DEEPGRAM_API_KEY not set - cannot auto-transcribe reference audio")
            return None
        
        logger.info(f"Transcribing reference audio with Deepgram: {audio_path}")
        
        # Read audio file
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        # Create Deepgram client and transcribe
        client = DeepgramClient(api_key)
        
        source = FileSource(buffer=audio_data)
        options = PrerecordedOptions(model="nova-2", language="en")
        
        response = client.listen.prerecorded.v("1").transcribe_file(source, options)
        
        # Extract transcript
        text = response.results.channels[0].alternatives[0].transcript.strip()
        
        if text:
            _ref_text_cache[audio_path] = text
            logger.info(f"Transcribed reference audio: {text[:100]}...")
            return text
        else:
            logger.warning(f"Transcription returned empty text for {audio_path}")
            return None
            
    except ImportError as e:
        logger.warning(f"Deepgram SDK not installed - cannot auto-transcribe reference audio: {e}")
        return None
    except Exception as e:
        logger.error(f"Error transcribing reference audio: {e}")
        return None


class NoReferenceAudioError(Exception):
    """Raised when CSM generation is attempted without reference audio."""
    pass


async def get_or_create_client(log_id: str, context=None) -> CSMClient:
    """Get or create CSM client for session."""
    if log_id not in _csm_clients:
        logger.info(f"Creating new CSM client for session {log_id}")
        client = CSMClient(CSM_SERVER_URL)
        await client.connect()
        
        # Get reference audio from persona or environment
        ref_audio = None
        ref_text = CSM_REF_TEXT
        
        # Try persona first
        voice_path = await get_voice_path_from_context(context)
        if voice_path:
            ref_audio = _load_ref_audio(voice_path)
            # Try to get ref_text from persona first
            persona_ref_text = await get_ref_text_from_context(context)
            if persona_ref_text:
                ref_text = persona_ref_text
            elif not ref_text:
                # Auto-transcribe if no ref_text provided
                transcribed = await transcribe_reference_audio(voice_path)
                if transcribed:
                    ref_text = transcribed
            logger.info(f"Loaded reference audio from persona: {voice_path}")
        elif CSM_REF_AUDIO and os.path.exists(CSM_REF_AUDIO):
            ref_audio = _load_ref_audio(CSM_REF_AUDIO)
            logger.info(f"Loaded reference audio from env: {CSM_REF_AUDIO}")
            # Auto-transcribe env ref audio if no ref_text
            if not ref_text:
                transcribed = await transcribe_reference_audio(CSM_REF_AUDIO)
                if transcribed:
                    ref_text = transcribed
        
        # CRITICAL: Fail if no reference audio - CSM will use wrong voice without it
        if ref_audio is None:
            raise NoReferenceAudioError(
                f"No reference audio found for CSM TTS! "
                f"Set persona voice_id to an audio file path, or set MR_CSM_REF_AUDIO env var. "
                f"Session: {log_id}"
            )
        
        await client.init_session(
            session_id=log_id,
            ref_audio=ref_audio,
            ref_text=ref_text,
            speaker_id=CSM_SPEAKER_ID
        )
        _csm_clients[log_id] = client
        
    return _csm_clients[log_id]


@pipe(name='sip_audio_in', priority=10)
async def forward_audio_to_csm(data: dict, context=None) -> dict:
    """
    Intercept incoming SIP audio and forward to CSM server.
    
    This pipe runs in parallel with Deepgram STT, streaming audio
    continuously so there's zero upload latency at generation time.
    """
    if not context or not hasattr(context, 'log_id'):
        return data
        
    audio_bytes = data.get('audio_bytes')
    if not audio_bytes:
        return data
        
    try:
        # Get or create client (non-blocking if already exists)
        if context.log_id in _csm_clients:
            client = _csm_clients[context.log_id]
            await client.send_audio(audio_bytes)
    except Exception as e:
        # Don't log every error - too noisy
        pass
        
    return data


@hook()
async def user_utterance_complete(text: str, context=None):
    """
    Called when Deepgram finalizes an utterance.
    Tell CSM server the user turn is complete with transcribed text.
    """
    if not context or not hasattr(context, 'log_id'):
        return
        
    try:
        client = _csm_clients.get(context.log_id)
        if client:
            await client.user_turn_end(text)
            logger.info(f"User turn complete: {text[:50]}...")
    except Exception as e:
        logger.error(f"Error sending user_turn_end to CSM: {e}")


@service()
async def stream_tts(
    text: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> AsyncGenerator[bytes, None]:
    """
    Stream TTS audio using CSM.
    
    Compatible with mr_eleven_stream interface.
    Yields ulaw 8kHz audio chunks (160 bytes = 20ms each).
    
    Args:
        text: Text to synthesize
        context: MindRoot context with log_id
        **kwargs: Additional parameters (ignored for compatibility)
    
    Yields:
        bytes: ulaw 8kHz audio chunks
    """
    stream_start = time.perf_counter()
    
    if not context or not hasattr(context, 'log_id'):
        logger.error("stream_tts requires context with log_id")
        return
        
    try:
        client = await get_or_create_client(context.log_id, context)
        
        chunk_count = 0
        async for audio_chunk in client.generate(text):
            chunk_count += 1
            print(f"STREAM_TTS: chunk #{chunk_count}, {len(audio_chunk)} bytes")
            elapsed = time.perf_counter() - stream_start
            logger.info(
                f"STREAM_TTS: yielding chunk #{chunk_count}, {len(audio_chunk)} bytes, "
                f"elapsed={elapsed*1000:.1f}ms"
            )
            yield audio_chunk
            
    except Exception as e:
        logger.error(f"Error in stream_tts: {e}")
        raise


@command()
async def speak(
    text: str,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Convert text to speech using CSM streaming TTS.
    
    This command streams the audio in real-time and is designed for backend
    integration with phone systems, audio pipelines, or other streaming audio consumers.
    
    Args:
        text: Text to speak
        context: MindRoot context
    
    Returns:
        None on success, error message on failure
    
    Example:
        { "speak": { "text": "Hello, this is a test message" } }
        { "speak": { "text": "How can I help you today?" } }
    """
    if not context or not hasattr(context, 'log_id'):
        logger.error("speak requires context with log_id")
        return "Error: speak requires context with log_id"
        
    log_id = context.log_id
    
    # Prevent concurrent speak() calls for same session
    if log_id not in _active_speak_locks:
        _active_speak_locks[log_id] = asyncio.Lock()
    
    lock = _active_speak_locks[log_id]
    
    if lock.locked():
        logger.warning(f"speak() already running for log_id {log_id}")
        return "ERROR: Speech already in progress"
    
    await lock.acquire()
    
    bypass_pacer = CSM_BYPASS_PACER
    
    speak_start = time.perf_counter()
    print(f"SPEAK: Starting speak() for text: {text[:50]}...")
    logger.info(f"SPEAK: Starting speak() for text: {text[:50]}...")
    
    try:
        # Check if SIP output is available
        sip_available = service_manager.functions.get('sip_audio_out_chunk') is not None
        
        if not sip_available:
            # Local playback mode - just generate and discard
            logger.info("SIP not available, generating audio without playback")
            async for chunk in stream_tts(text=text, context=context):
                pass
            return None
        
        # Check if audio is halted (interrupted state)
        try:
            is_halted = await service_manager.sip_is_audio_halted(context=context)
            if is_halted:
                logger.info("Audio halted, skipping speak command")
                return None
        except:
            pass
        
        chunk_count = 0
        
        if bypass_pacer:
            # BYPASS MODE: Send directly to SIP without pacing
            print("SPEAK: BYPASS_PACER mode - sending directly to SIP")
            logger.info("SPEAK: BYPASS_PACER mode - sending directly to SIP")
            
            async for audio_chunk in stream_tts(text=text, context=context):
                chunk_count += 1
                if chunk_count == 1:
                    first_chunk_time = time.perf_counter()
                    logger.info(f"SPEAK_BYPASS: First chunk, sending directly to SIP, elapsed={((first_chunk_time - speak_start)*1000):.1f}ms")
                    print(f"SPEAK_BYPASS: First chunk, elapsed={((first_chunk_time - speak_start)*1000):.1f}ms")
                
                # Send directly to SIP
                try:
                    result = await service_manager.sip_audio_out_chunk(
                        audio_chunk, timestamp=None, context=context
                    )
                    if result is False:
                        logger.warning("SPEAK_BYPASS: sip_audio_out_chunk returned False, stopping")
                        break
                except Exception as e:
                    logger.error(f"SPEAK_BYPASS: Error sending to SIP: {e}")
                    break
            
            finished_time = time.perf_counter()
            logger.info(f"SPEAK_BYPASS: All {chunk_count} chunks sent, elapsed={((finished_time - speak_start)*1000):.1f}ms")
            print(f"SPEAK_BYPASS: All {chunk_count} chunks sent, elapsed={((finished_time - speak_start)*1000):.1f}ms")
            
        else:
            # NORMAL MODE: Use AudioPacer
            pacer = AudioPacer(sample_rate=8000)
            
            async def send_to_sip(chunk, timestamp=None, context=None):
                """Callback for AudioPacer to send chunks to SIP."""
                try:
                    logger.debug(f"SPEAK: send_to_sip called with {len(chunk)} bytes")
                    result = await service_manager.sip_audio_out_chunk(
                        chunk, timestamp=timestamp, context=context
                    )
                    return result
                except Exception as e:
                    logger.error(f"Error sending to SIP: {e}")
                    return False
            
            await pacer.start_pacing(send_to_sip, context)
            pacer_started = time.perf_counter()
            logger.info(f"SPEAK: Pacer started, elapsed={((pacer_started - speak_start)*1000):.1f}ms")
            
            try:
                async for audio_chunk in stream_tts(text=text, context=context):
                    if pacer.interrupted:
                        logger.debug("Pacer interrupted, stopping chunk buffering")
                        break
                    
                    await pacer.add_chunk(audio_chunk)
                    chunk_count += 1
                    if chunk_count == 1:
                        first_chunk_time = time.perf_counter()
                        logger.info(f"SPEAK: First chunk added to pacer, elapsed={((first_chunk_time - speak_start)*1000):.1f}ms")
                    
                pacer.mark_finished()
                finished_time = time.perf_counter()
                logger.info(f"SPEAK: All {chunk_count} chunks added, marking finished, elapsed={((finished_time - speak_start)*1000):.1f}ms")
                
                if not pacer.interrupted:
                    logger.debug(f"All {chunk_count} chunks buffered, waiting for pacer...")
                    await pacer.wait_until_done()
                    logger.info(f"SPEAK: Pacer done, total elapsed={((time.perf_counter() - speak_start)*1000):.1f}ms")
                    
            finally:
                await pacer.stop()
            
            if pacer.interrupted:
                logger.info(f"Interrupted after {chunk_count} chunks, {pacer.bytes_sent} bytes sent")
                if chunk_count < 2:
                    return "SYSTEM: WARNING - Command interrupted!\n\n"
                return None
        
        logger.info(f"SPEAK: Completed {chunk_count} chunks")
        
        return None
        
    except Exception as e:
        logger.error(f"Error in speak command: {e}")
        return f"Error: {str(e)}"
        
    finally:
        if lock.locked():
            lock.release()


@hook()
async def on_interrupt(context=None):
    """Handle user interruption - stop CSM generation."""
    if not context or not hasattr(context, 'log_id'):
        return
        
    client = _csm_clients.get(context.log_id)
    if client:
        await client.interrupt()
        logger.info(f"Interrupted CSM generation for {context.log_id}")


@hook()
async def session_end(context=None):
    """Clean up CSM client when session ends."""
    if not context or not hasattr(context, 'log_id'):
        return
        
    log_id = context.log_id
    
    client = _csm_clients.pop(log_id, None)
    if client:
        await client.close()
        logger.info(f"Closed CSM session for {log_id}")
    
    # Clean up lock
    _active_speak_locks.pop(log_id, None)
