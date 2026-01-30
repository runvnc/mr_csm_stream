"""
CSM WebSocket Client

Handles communication with the CSM server for streaming TTS.
Handles communication with the CSM server for streaming TTS.
"""

import asyncio
import base64
import json
import logging
from typing import AsyncGenerator, Optional
import socket
import time

import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


class CSMClient:
    """WebSocket client for CSM server."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.session_id: Optional[str] = None
        self._generating = False
        self._connected = False
        self._reconnect_lock = asyncio.Lock()
        
    async def connect(self):
        """Connect to CSM server."""
        try:
            self.ws = await websockets.connect(
                self.server_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5
            )
            
            # Disable Nagle's algorithm for lower latency
            try:
                sock = self.ws.transport.get_extra_info('socket')
                if sock:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    logger.info("TCP_NODELAY enabled on CSM WebSocket client")
            except Exception as e:
                logger.warning(f"Could not set TCP_NODELAY on client: {e}")
            self._connected = True
            logger.info(f"Connected to CSM server: {self.server_url}")
        except Exception as e:
            logger.error(f"Failed to connect to CSM server: {e}")
            raise
            
    async def ensure_connected(self):
        """Ensure connection is active, reconnect if needed."""
        async with self._reconnect_lock:
            if not self._connected or self.ws is None or self.ws.closed:
                logger.info("Reconnecting to CSM server...")
                await self.connect()
                if self.session_id:
                    # Re-initialize session after reconnect
                    await self._send_init(self.session_id)
        
    async def _send_init(self, session_id: str):
        """Send init message (internal)."""
        msg = {
            "type": "init",
            "session_id": session_id,
            "speaker_id": 0
        }
        await self.ws.send(json.dumps(msg))
        # Wait for ready response
        response = await self.ws.recv()
        data = json.loads(response)
        if data.get("type") != "ready":
            logger.warning(f"Unexpected init response: {data}")
        
    async def init_session(
        self,
        session_id: str,
        ref_audio: Optional[bytes] = None,
        ref_text: str = "",
        speaker_id: int = 0
    ):
        """Initialize session with reference audio."""
        await self.ensure_connected()
        
        self.session_id = session_id
        
        msg = {
            "type": "init",
            "session_id": session_id,
            "speaker_id": speaker_id
        }
        
        if ref_audio:
            msg["ref_audio_base64"] = base64.b64encode(ref_audio).decode()
            msg["ref_text"] = ref_text
            
        await self.ws.send(json.dumps(msg))
        
        # Wait for ready response
        response = await self.ws.recv()
        data = json.loads(response)
        if data.get("type") == "ready":
            logger.info(f"Initialized CSM session: {session_id}")
        else:
            logger.warning(f"Unexpected init response: {data}")
        
    async def send_audio(self, audio_bytes: bytes):
        """Send audio chunk to server."""
        if not self._connected or self.ws is None:
            return
            
        try:
            msg = {
                "type": "audio",
                "data": base64.b64encode(audio_bytes).decode()
            }
            await self.ws.send(json.dumps(msg))
        except ConnectionClosed:
            self._connected = False
            logger.warning("Connection closed while sending audio")
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
        
    async def user_turn_end(self, text: str):
        """Signal end of user turn with transcribed text."""
        await self.ensure_connected()
        
        msg = {
            "type": "user_turn_end",
            "text": text
        }
        await self.ws.send(json.dumps(msg))
        # Do not block waiting for server acknowledgment.
        # The server may still send {"type":"user_turn_processed"}; we'll ignore it if received later.
        
    async def generate(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate audio and yield chunks."""
        await self.ensure_connected()
        
        gen_start = time.perf_counter()
        chunk_count = 0
        
        self._generating = True
        
        msg = {
            "type": "generate",
            "text": text
        }
        await self.ws.send(json.dumps(msg))
        logger.info(f"Generating: {text[:50]}...")
        
        try:
            while self._generating:
                try:
                    response = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                    recv_time = time.perf_counter()
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for audio chunk")
                    break
                    
                data = json.loads(response)
                
                if data["type"] == "audio":
                    audio_bytes = base64.b64decode(data["data"])
                    chunk_count += 1
                    elapsed = recv_time - gen_start
                    logger.info(
                        f"CSM_CLIENT_RECV: chunk #{chunk_count}, {len(audio_bytes)} bytes, "
                        f"elapsed={elapsed*1000:.1f}ms"
                    )
                    yield audio_bytes
                    asyncio.sleep(0.02)  # Yield control to event loop

                elif data["type"] == "done":
                    logger.info("Generation complete")
                    break
                    
                elif data["type"] == "error":
                    logger.error(f"CSM error: {data.get('message')}")
                    break
                    
                elif data["type"] == "interrupted":
                    logger.info("Generation interrupted by server")
                    break

                elif data["type"] == "user_turn_processed":
                    # Ack from a prior user_turn_end; safe to ignore here.
                    continue
                    
        except ConnectionClosed:
            self._connected = False
            logger.warning("Connection closed during generation")
        except Exception as e:
            logger.error(f"Error during generation: {e}")
        finally:
            self._generating = False
            
    async def interrupt(self):
        """Interrupt current generation."""
        self._generating = False
        
        if self._connected and self.ws:
            try:
                msg = {"type": "interrupt"}
                await self.ws.send(json.dumps(msg))
                logger.info("Sent interrupt to CSM server")
            except Exception as e:
                logger.error(f"Error sending interrupt: {e}")
            
    async def close(self):
        """Close connection."""
        self._generating = False
        self._connected = False
        
        if self.ws:
            try:
                msg = {"type": "close"}
                await self.ws.send(json.dumps(msg))
            except:
                pass
            try:
                await self.ws.close()
            except:
                pass
            self.ws = None
            logger.info(f"Closed CSM session: {self.session_id}")
