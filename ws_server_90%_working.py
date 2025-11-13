# ws_server.py — Twilio TwiML + recording webhook + hold loop + REALTIME STREAM (FastAPI)

import os
import asyncio
import json
import logging
import time
import urllib.parse
from typing import Optional, Dict

from fastapi import FastAPI, Request, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, PlainTextResponse
from twilio.twiml.voice_response import VoiceResponse
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ws_server")

# -------- Env / Config --------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Prefer your key name, fallback to standard
OPENAI_API_KEY = os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

# Optional (kept from your file, still usable by legacy flow)
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
ELEVEN_VOICE = os.getenv("ELEVEN_VOICE", "Xb7hH8MSUJpSbSDYk0k2")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION", "eu-north-1")
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM", "+15312303465")

if not OPENAI_API_KEY:
    logger.warning("No OpenAI key found. Set OpenAI_Key or OPENAI_API_KEY.")
if not PUBLIC_BASE_URL:
    logger.warning("PUBLIC_BASE_URL is not set; realtime Twilio <Stream> will fail.")

def _has_openai_key() -> bool:
    """Runtime check used by endpoints (startup warning above is not enough)."""
    return bool(os.getenv("OPENAI_KEY") or os.getenv("OPENAI_API_KEY"))

def _ws_host_from_public_base() -> Optional[str]:
    """Render-safe: always derive WS host from PUBLIC_BASE_URL, not request.host."""
    base = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
    if not base:
        return None
    return base.replace("https://", "").replace("http://", "")

app = FastAPI()

# ----------- Legacy turn-based state (kept from your flow) -----------
_state: Dict[str, Dict] = {}
_state_lock = asyncio.Lock()

MIN_DURATION_SECONDS = 2
RECORDING_COOLDOWN_SECONDS = 2.0

async def should_ignore_recording(call_sid: str, rec_sid: Optional[str], rec_duration: Optional[str]) -> bool:
    now = time.time()
    async with _state_lock:
        st = _state.get(call_sid, {"last_rec_sid": None, "last_at": 0.0, "turn": 0})
        if rec_duration is not None:
            try:
                if int(rec_duration) < MIN_DURATION_SECONDS:
                    logger.info("[%s] ignore: too short (%ss)", call_sid, rec_duration)
                    return True
            except Exception:
                pass
        if rec_sid and st.get("last_rec_sid") == rec_sid:
            logger.info("[%s] ignore: duplicate RecordingSid=%s", call_sid, rec_sid)
            return True
        if now - float(st.get("last_at", 0.0)) < RECORDING_COOLDOWN_SECONDS:
            logger.info("[%s] ignore: cooldown not elapsed", call_sid)
            return True
        if rec_sid:
            st["last_rec_sid"] = rec_sid
        st["last_at"] = now
        _state[call_sid] = st
        return False

def twiml(resp: VoiceResponse) -> Response:
    return Response(content=str(resp), media_type="text/xml")

def recording_callback_url(request: Request) -> str:
    return f"{str(request.base_url).rstrip('/')}/recording"

_hold: Dict[str, Dict] = {}
_hold_lock = asyncio.Lock()

# -------------------- Health / Root --------------------
@app.get("/")
async def root():
    return PlainTextResponse(
        "ok: /twiml (turn-based), /twiml_stream (realtime), /health",
        status_code=200
    )

@app.get("/health")
async def health():
    return PlainTextResponse("ok", status_code=200)

# -------------------- Legacy turn-based TwiML --------------------
@app.get("/twiml")
@app.post("/twiml")
async def twiml_entry(request: Request):
    try:
        vr = VoiceResponse()
        vr.say("Hello! Please leave your message after the beep.", voice="alice")
        vr.record(max_length=30, play_beep=True, timeout=3, action=recording_callback_url(request))
        return twiml(vr)
    except Exception as e:
        logger.exception("twiml_entry error: %s", e)
        vr = VoiceResponse()
        vr.say("An application error has occurred. Goodbye.", voice="alice")
        vr.hangup()
        return twiml(vr)

@app.options("/recording")
async def recording_options():
    return PlainTextResponse("", status_code=200)

@app.post("/recording")
@app.get("/recording")
async def recording_webhook(
    request: Request,
    CallSid: Optional[str] = Form(None),
    From: Optional[str] = Form(None),
    RecordingUrl: Optional[str] = Form(None),
    RecordingSid: Optional[str] = Form(None),
    RecordingDuration: Optional[str] = Form(None),
    q_CallSid: Optional[str] = Query(None, alias="CallSid"),
    q_From: Optional[str] = Query(None, alias="From"),
    q_RecordingUrl: Optional[str] = Query(None, alias="RecordingUrl"),
    q_RecordingSid: Optional[str] = Query(None, alias="RecordingSid"),
    q_RecordingDuration: Optional[str] = Query(None, alias="RecordingDuration"),
):
    call_sid = CallSid or q_CallSid
    from_num = From or q_From
    rec_url  = RecordingUrl or q_RecordingUrl
    rec_sid  = RecordingSid or q_RecordingSid
    rec_dur  = RecordingDuration or q_RecordingDuration

    if not call_sid or not rec_url:
        vr = VoiceResponse()
        vr.say("We did not get your message. Please try again.", voice="alice")
        vr.record(max_length=30, play_beep=True, timeout=3, action=recording_callback_url(request))
        return twiml(vr)

    if await should_ignore_recording(call_sid, rec_sid, rec_dur):
        vr = VoiceResponse()
        vr.pause(length=1)
        vr.record(max_length=30, play_beep=True, timeout=3, action=recording_callback_url(request))
        return twiml(vr)

    asyncio.create_task(process_recording_background(call_sid, rec_url, from_num))

    vr = VoiceResponse()
    vr.say("Got it. Please hold while I prepare your response.", voice="alice")
    hold_url = f"{str(request.base_url).rstrip('/')}/hold?convo_id={urllib.parse.quote_plus(call_sid)}"
    vr.redirect(hold_url)
    return twiml(vr)

@app.post("/_test_set_hold")
async def test_set_hold(convo_id: str = Query(...), reply_text: str = Query("Hello from test"), tts_url: Optional[str] = Query(None)):
    async with _hold_lock:
        _hold[convo_id] = {"reply_text": reply_text, "tts_url": tts_url}
    return {"ok": True, "convo_id": convo_id, "payload": _hold[convo_id]}

@app.get("/hold")
async def hold(request: Request, convo_id: str = Query(...)):
    try:
        async with _hold_lock:
            payload = _hold.pop(convo_id, None)

        if payload:
            reply_text = payload.get("reply_text") or "Here is your reply."
            tts_url = payload.get("tts_url")
            async with _state_lock:
                st = _state.get(convo_id, {"turn": 0, "last_at": time.time()})
                st["turn"] = int(st.get("turn", 0)) + 1
                st["last_at"] = time.time()
                _state[convo_id] = st

            vr = VoiceResponse()
            if tts_url:
                vr.play(tts_url)
            else:
                vr.say(reply_text, voice="alice")
            vr.record(max_length=30, play_beep=True, timeout=3, action=recording_callback_url(request))
            return twiml(vr)

        vr = VoiceResponse()
        vr.pause(length=2)
        vr.redirect(f"{str(request.base_url).rstrip('/')}/hold?convo_id={urllib.parse.quote_plus(convo_id)}")
        return twiml(vr)
    except Exception as e:
        logger.exception("Hold error: %s", e)
        vr = VoiceResponse()
        vr.say("An error occurred.", voice="alice")
        return twiml(vr)

# ----------- Background (legacy) stub kept -----------
import re, requests
from requests.auth import HTTPBasicAuth
_TWILIO_REC_RE = re.compile(r"https://api\.twilio\.com/2010-04-01/Accounts/[^/]+/Recordings/(RE[a-zA-Z0-9]+)(?:\.(mp3|wav))?$")

def build_download_url(url: str) -> str:
    m = _TWILIO_REC_RE.match(url)
    if m and not m.group(2):
        return f"{url}.mp3"
    return url

def download_bytes_with_retry(url: str, auth=None, timeout=20, attempts=3, backoff=0.6, min_size_bytes: int = 1024) -> bytes:
    last_exc = None
    for i in range(1, attempts + 1):
        try:
            r = requests.get(url, auth=auth, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            if len(r.content) < min_size_bytes:
                raise RuntimeError("Downloaded content too small")
            return r.content
        except Exception as e:
            last_exc = e
            time.sleep(backoff * i)
    raise last_exc or RuntimeError("download failed")

async def process_recording_background(call_sid: str, recording_url: str, from_number: Optional[str] = None):
    # Keep your old transcription + reply + (optional TTS to S3) implementation if you still use the legacy route
    pass

# =======================================================================
# ===============  REALTIME STREAMING WITH BARGE-IN  ====================
# =======================================================================

from aiohttp import ClientSession, WSMsgType

@app.post("/twiml_stream")
@app.get("/twiml_stream")
async def twiml_stream(request: Request):
    vr = VoiceResponse()

    # Hard fail (speak + hang up) if config is missing, to avoid “welcome then drop”.
    if not _has_openai_key():
        logger.error("Realtime requested but no OpenAI key set. Set OpenAI_Key or OPENAI_API_KEY.")
        vr.say("Configuration error. Open A I key is missing. Please try again later.", voice="alice")
        vr.hangup()
        return twiml(vr)

    ws_host = _ws_host_from_public_base()
    if not ws_host:
        logger.error("Realtime requested but PUBLIC_BASE_URL is not set.")
        vr.say("Configuration error. Public base URL is not set. Goodbye.", voice="alice")
        vr.hangup()
        return twiml(vr)

    vr.say("You are connected to the real time assistant.", voice="alice")
    # Always use PUBLIC_BASE_URL host for Twilio <Stream> (Render-friendly)
    vr.connect().stream(url=f"wss://{ws_host}/twilio-media")
    return twiml(vr)

# How much audio to accumulate before committing (in bytes)
# ~120 ms @ 8kHz mulaw = 0.12 * 8000 * 1 = 960 bytes
COMMIT_BYTES_TARGET = 960

# Track per-stream state
stream_state: Dict[str, Dict[str, int]] = {}  # {stream_sid: {"accum_bytes": int}}

@app.websocket("/twilio-media")
async def twilio_media_ws(ws: WebSocket):
    await ws.accept()
    call_info = {"stream_sid": None}
    logger.info("Twilio WS connected")

    # OpenAI Realtime WS client session
    openai_ws = None
    openai_task = None

    async def openai_connect_and_pump():
        nonlocal openai_ws
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not set — aborting realtime session.")
            return
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }
        # Update model string as needed to the newest realtime-preview
        url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
        async with ClientSession() as session:
            async with session.ws_connect(url, headers=headers) as ows:
                openai_ws = ows

                # Configure session audio formats to match Twilio (μ-law 8k)
                await ows.send_json({
                    "type": "session.update",
                    "session": {
                        "input_audio_format": "g711_ulaw",
                        "output_audio_format": "g711_ulaw",
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 200,
                            "silence_duration_ms": 600
                        },
                        "voice": "verse",
                        "instructions": "You are a concise helpful voice agent. Keep answers short.",
                    }
                })

                # Pump OpenAI -> Twilio
                async for msg in ows:
                    if msg.type == WSMsgType.TEXT:
                        evt = msg.json()
                        etype = evt.get("type")

                        if etype == "response.audio.delta":
                            chunk_b64 = evt.get("delta")
                            if chunk_b64 and ws.client_state.name != "DISCONNECTED":
                                # Forward model audio to Twilio stream
                                await ws.send_text(json.dumps({
                                    "event": "media",
                                    "streamSid": call_info.get("stream_sid"),
                                    "media": {"payload": chunk_b64}
                                }))

                        elif etype == "response.completed":
                            # One answer finished; next user audio will trigger new response
                            pass

                        elif etype == "error":
                            logger.error("OpenAI error event: %s", evt)
                            break

                    elif msg.type == WSMsgType.BINARY:
                        # Not used
                        pass
                    elif msg.type == WSMsgType.ERROR:
                        logger.error("OpenAI ws error")
                        break

    async def ensure_openai():
        nonlocal openai_task
        if openai_task is None or openai_task.done():
            openai_task = asyncio.create_task(openai_connect_and_pump())

    try:
        await ensure_openai()

        while True:
            raw = await ws.receive_text()
            frame = json.loads(raw)
            event = frame.get("event")

            if event == "start":
                call_info["stream_sid"] = frame["start"]["streamSid"]
                logger.info("Stream started: %s", call_info["stream_sid"])
                stream_state[call_info["stream_sid"]] = {"accum_bytes": 0}

            elif event == "media":
                # Incoming μ-law 8k audio from Twilio (base64)
                payload_b64 = frame["media"]["payload"]
                if not payload_b64:
                    # empty frame; ignore
                    continue

                await ensure_openai()
                if openai_ws is not None and not openai_ws.closed:
                    # Append audio & commit; model VAD will decide when to respond
                    await openai_ws.send_json({"type": "input_audio_buffer.append", "audio": payload_b64})

                     # accumulate exact byte-count (base64-decoded length)
                    try:
                        raw_len = len(base64.b64decode(payload_b64))
                    except Exception:
                    # if decoding fails, fall back to an estimate (not typical)
                        raw_len = 0

                    sid = call_info.get("stream_sid") or "unknown"
                    if sid not in stream_state:
                        stream_state[sid] = {"accum_bytes": 0}
                        stream_state[sid]["accum_bytes"] += raw_len
                    if stream_state[sid]["accum_bytes"] >= COMMIT_BYTES_TARGET:    
                        await openai_ws.send_json({"type": "input_audio_buffer.commit"})
                        await openai_ws.send_json({"type": "response.create", "response": {"modalities": ["text", "audio"]}})
                        stream_state[sid]["accum_bytes"] = 0

            elif event == "mark":
                # Optional markers
                pass

            elif event == "stop":
                logger.info("Stream stopped by Twilio.")
                sid = call_info.get("stream_sid")
                if sid and sid in stream_state:
                    stream_state.pop(sid, None)
                break
            # --- Hard barge-in hook (optional) ---
            # If you want to force-cut current speech on any new user audio:
            # if event == "media" and openai_ws is not None and not openai_ws.closed:
            #     await openai_ws.send_json({"type": "response.cancel"})

    except WebSocketDisconnect:
        logger.info("Twilio WS disconnected")
    except Exception as e:
        logger.exception("Twilio WS error: %s", e)
    finally:
        try:
            if openai_task:
                openai_task.cancel()
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
