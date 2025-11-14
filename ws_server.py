"""
ws_server.py — Exotel Outbound Realtime Voicebot + OpenAI Realtime
------------------------------------------------------------------
Features:
- Trigger outbound calls to an Exotel Voicebot Flow (Connect API v1)
- Serve /exotel-ws-bootstrap for Exotel Voicebot (Bidirectional) applet
- Handle /exotel-media WebSocket for realtime AI voicebot (OpenAI Realtime)
- Receive Exotel call status webhooks at /exotel/status
- CSV / batch outbound helpers

ENV (set in Render):
  EXO_SID, EXO_API_KEY, EXO_API_TOKEN, EXO_FLOW_ID, EXO_SUBDOMAIN=api, EXO_CALLER_ID
  OPENAI_API_KEY or OpenAI_Key or OPENAI_KEY
  OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview
  PUBLIC_BASE_URL=<your-render-hostname-without-https> e.g. openai-exotel-elevenlabs-realtime.onrender.com
  LOG_LEVEL=INFO
  SAVE_TTS_WAV=1 (optional, save OpenAI audio to /tmp)
"""

import os, json, base64, asyncio, logging, time, wave, audioop, csv
from typing import Optional, List

import httpx
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Body,
    UploadFile,
    File,
    Request,
    HTTPException,
)
from fastapi.responses import PlainTextResponse, JSONResponse
from aiohttp import ClientSession, WSMsgType
from pydantic import BaseModel


# ---------------- Logging ----------------
level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, level, logging.INFO))
logger = logging.getLogger("ws_server")

# ---------------- FastAPI ----------------
app = FastAPI(title="Exotel Outbound Realtime Voicebot")


# ---------------- Request models ----------------
class OutboundCallRequest(BaseModel):
    """Body for /exotel-outbound-call"""
    to_number: str   # customer mobile/landline, e.g. "09052161119"


# ---------------- Exotel ENV ----------------
EXO_SID       = os.getenv("EXO_SID", "")
EXO_API_KEY   = os.getenv("EXO_API_KEY", "")
EXO_API_TOKEN = os.getenv("EXO_API_TOKEN", "")
EXO_FLOW_ID   = os.getenv("EXO_FLOW_ID", "")            # App / Flow app id (e.g. 1075544)
EXO_SUBDOMAIN = os.getenv("EXO_SUBDOMAIN", "api")       # "api" or "api.in"
EXO_CALLER_ID = os.getenv("EXO_CALLER_ID", "")          # Your Exophone


# ---------------- OpenAI ENV ----------------
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OpenAI_Key")
    or os.getenv("OPENAI_KEY", "")
)
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")

# ---------------- Misc ENV ----------------
PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").strip()  # no protocol
SAVE_TTS_WAV    = os.getenv("SAVE_TTS_WAV", "0") == "1"


# ---------------- Helper: downsample ----------------
def downsample_24k_to_8k_pcm16(pcm24: bytes) -> bytes:
    """24 kHz mono PCM16 -> 8 kHz mono PCM16 using stdlib audioop."""
    converted, _ = audioop.ratecv(pcm24, 2, 1, 24000, 8000, None)
    return converted


# ---------------- Helper: Exotel outbound (Connect API) ----------------
async def exotel_connect_voicebot(to_e164: str) -> dict:
    """
    Start an outbound call via Exotel Connect API and drop the callee
    into your Voicebot Flow (EXO_FLOW_ID) which points to /exotel-ws-bootstrap.
    """
    missing = [
        name for name, value in [
            ("EXO_SID", EXO_SID),
            ("EXO_API_KEY", EXO_API_KEY),
            ("EXO_API_TOKEN", EXO_API_TOKEN),
            ("EXO_FLOW_ID", EXO_FLOW_ID),
            ("EXO_CALLER_ID", EXO_CALLER_ID),
        ] if not value
    ]
    if missing:
        msg = f"Missing Exotel env vars: {', '.join(missing)}"
        logger.error(msg)
        raise HTTPException(status_code=500, detail=msg)

    # Connect API endpoint
    base = f"https://{EXO_SUBDOMAIN}.exotel.com"
    url = f"{base}/v1/Accounts/{EXO_SID}/Calls/connect.json"

    # This App / Flow should contain the Voicebot applet pointing to /exotel-ws-bootstrap
    exoml_url = f"https://my.exotel.com/{EXO_SID}/exoml/start_voice/{EXO_FLOW_ID}"

    payload = {
        "From": to_e164,          # customer number to call first
        "CallerId": EXO_CALLER_ID,
        "Url": exoml_url,
        "CallType": "trans",
    }

    logger.info("Exotel Connect: %s -> %s (Url=%s)", EXO_CALLER_ID, to_e164, exoml_url)

    async with httpx.AsyncClient(timeout=20.0, auth=(EXO_API_KEY, EXO_API_TOKEN)) as client:
        resp = await client.post(url, data=payload)
        text = resp.text

    if resp.status_code >= 400:
        logger.error("Exotel outbound error %s: %s", resp.status_code, text)
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Exotel error ({resp.status_code}): {text}",
        )

    try:
        data = resp.json()
    except Exception:
        data = {"raw": text}

    logger.info("Exotel outbound accepted: %s", data)
    return data


# ---------------- Health ----------------
@app.get("/health", response_class=PlainTextResponse)
async def health():
    return "ok"


# ---------------- Outbound REST (single) ----------------
@app.post("/exotel-outbound-call")
async def exotel_outbound_call(body: OutboundCallRequest):
    """
    Start an outbound call to a customer and connect them to your
    realtime OpenAI bot via the Exotel Flow / App (EXO_FLOW_ID).

    Flow:
    1) Exotel calls `body.to_number` from your EXO_CALLER_ID.
    2) When the customer answers, Exotel enters the Voice app 1075544.
    3) That app has a Voicebot node pointing to your /exotel-ws-bootstrap.
    4) Exotel connects its websocket to /exotel-media, which talks to OpenAI.
    """
    # Normalize to E.164 (+91...) if you want India default
    to = body.to_number
    if not to.startswith("+"):
        to = f"+91{to}"

    res = await exotel_connect_voicebot(to)
    return {"status": "ok", "exotel": res}


# ---------------- Outbound REST (batch) ----------------
@app.post("/outbound/batch")
async def outbound_batch(numbers: List[str]):
    """
    POST /outbound/batch
    Body: ["9876543210", "9820098200", ...]
    Triggers sequential outbound realtime calls to a list of numbers.
    """
    results = []
    for n in numbers:
        try:
            to = n if n.startswith("+") else f"+91{n}"
            res = await exotel_connect_voicebot(to)
            results.append({"number": n, "status": "ok", "exotel": res})
            await asyncio.sleep(0.5)  # respect channel capacity & API rate limits
        except Exception as e:
            logger.exception("Error calling %s: %s", n, e)
            results.append({"number": n, "status": "error", "error": str(e)})
    return results


# ---------------- Outbound CSV uploader ----------------
@app.post("/outbound/csv")
async def outbound_csv(file: UploadFile = File(...)):
    """
    POST /outbound/csv
    Multipart form-data with a file field named 'file'.
    CSV format: number,name
      - 'number' column mandatory
      - 'name' optional
    Example:
      9876543210,Raj
      9820098200,Seema
    """
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    reader = csv.DictReader(text.splitlines())
    results = []

    for row in reader:
        number = (row.get("number") or "").strip()
        name = (row.get("name") or "").strip()
        if not number:
            continue
        try:
            to = number if number.startswith("+") else f"+91{number}"
            res = await exotel_connect_voicebot(to)
            results.append({"number": number, "name": name, "status": "ok", "exotel": res})
            await asyncio.sleep(0.5)
        except Exception as e:
            logger.exception("Error calling %s: %s", number, e)
            results.append({"number": number, "name": name, "status": "error", "error": str(e)})
    return {"count": len(results), "results": results}


# ---------------- Exotel status webhook ----------------
@app.post("/exotel/status")
async def exotel_status(request: Request):
    """
    Exotel call status webhook.
    Configure in Exotel dashboard to POST here.
    Logs call lifecycle events (answered, completed, failed, etc.).
    """
    # Exotel usually sends application/x-www-form-urlencoded
    try:
        form = await request.form()
        data = dict(form)
    except Exception:
        data = {}

    # Common fields: CallSid, Status, Direction, From, To, StartTime, EndTime, etc.
    call_sid = data.get("CallSid") or data.get("CallSid[]")
    status   = data.get("Status") or data.get("Status[]")
    frm      = data.get("From") or data.get("From[]")
    to       = data.get("To") or data.get("To[]")
    logger.info(
        "Exotel status: CallSid=%s Status=%s From=%s To=%s Raw=%s",
        call_sid,
        status,
        frm,
        to,
        data,
    )

    # You could persist this to DB here; for now just acknowledge
    return JSONResponse({"ok": True})


# ---------------- Bootstrap for Exotel Voicebot ----------------
@app.get("/exotel-ws-bootstrap")
async def exotel_ws_bootstrap():
    """
    Exotel Voicebot applet hits this URL and expects: {"url": "wss://<host>/exotel-media"}.
    """
    try:
        base = PUBLIC_BASE_URL or "openai-exotel-elevenlabs-realtime.onrender.com"
        url = f"wss://{base}/exotel-media"
        logger.info("Bootstrap served: %s", url)
        return {"url": url}
    except Exception as e:
        logger.exception("/exotel-ws-bootstrap error: %s", e)
        return {"url": f"wss://{(PUBLIC_BASE_URL or 'openai-exotel-elevenlabs-realtime.onrender.com')}/exotel-media"}


# ---------------- Realtime media bridge (Exotel <-> OpenAI) ----------------
@app.websocket("/exotel-media")
async def exotel_media_ws(ws: WebSocket):
    """
    Bidirectional audio bridge for Exotel Voicebot (callee leg @ 8 kHz PCM16).
    - Receives Exotel media events (8k PCM16 base64)
    - Sends inline input_audio turns to OpenAI Realtime (no buffer/commit)
    - Streams OpenAI audio deltas back to Exotel, downsampled 24k -> 8k
    - Supports barge-in
    """
    await ws.accept()
    logger.info("Exotel WS connected")

    if not OPENAI_API_KEY:
        logger.error("No OPENAI_API_KEY/OpenAI_Key configured; closing")
        await ws.close()
        return

    EXO_SR = 8000
    BYTES_PER_SAMPLE = 2
    MIN_WINDOW = int(EXO_SR * BYTES_PER_SAMPLE * 0.12)  # ~120ms =~ 1920 bytes @ 8k

    pending = False
    speaking = False
    connected_to_openai = False

    live_chunks: List[str] = []
    live_bytes = 0
    live_frames = 0

    barge_chunks: List[str] = []
    barge_bytes = 0
    barge_frames = 0

    openai_session: Optional[ClientSession] = None
    openai_ws = None
    pump_task: Optional[asyncio.Task] = None

    async def send_openai(payload: dict):
        if openai_ws is None or openai_ws.closed:
            logger.info("drop %s: OpenAI ws not ready/closed", payload.get("type"))
            return
        t = payload.get("type")
        if t != "response.audio.delta":
            logger.info("SENDING to OpenAI: %s", t)
        await openai_ws.send_json(payload)

    async def openai_connect():
        nonlocal openai_session, openai_ws, pump_task, connected_to_openai, pending, speaking
        if connected_to_openai:
            return
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "OpenAI-Beta": "realtime=v1"}
        url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

        openai_session = ClientSession()
        openai_ws = await openai_session.ws_connect(url, headers=headers)

        await send_openai({
            "type": "session.update",
            "session": {
                "input_audio_format":  "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": 600
                },
                "voice": "verse",
                "instructions": "You are a concise helpful voice agent. Reply in clear Indian English."
            }
        })

        async def pump_openai_to_exotel():
            nonlocal speaking, pending
            tts_dump: bytearray = bytearray()
            try:
                async for msg in openai_ws:
                    if msg.type == WSMsgType.TEXT:
                        evt = msg.json()
                        et = evt.get("type")

                        if et in ("response.output_audio.delta", "response.audio.delta"):
                            b64 = evt.get("delta")
                            if b64 and ws.client_state.name != "DISCONNECTED":
                                pcm24 = base64.b64decode(b64)
                                if SAVE_TTS_WAV:
                                    tts_dump.extend(pcm24)
                                pcm8 = downsample_24k_to_8k_pcm16(pcm24)
                                out_b64 = base64.b64encode(pcm8).decode("ascii")

                                speaking = True
                                await ws.send_text(json.dumps({"event": "media", "audio": out_b64}))

                        elif et == "response.completed":
                            speaking = False
                            pending = False
                            logger.info("OpenAI: response.completed")

                        elif et == "error":
                            logger.error("OpenAI error: %s", evt)
                            pending = False
                            break

                    elif msg.type == WSMsgType.ERROR:
                        logger.error("OpenAI ws error")
                        pending = False
                        break
            except Exception as e:
                logger.exception("OpenAI pump error: %s", e)
                pending = False
            finally:
                if SAVE_TTS_WAV and tts_dump:
                    fname = f"/tmp/openai_tts_{int(time.time())}.wav"
                    with wave.open(fname, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        wf.writeframes(bytes(tts_dump))
                    logger.info("Saved OpenAI TTS to %s", fname)

        pump_task = asyncio.create_task(pump_openai_to_exotel())
        connected_to_openai = True
        logger.info("OpenAI realtime connected")

    async def openai_close():
        try:
            if pump_task and not pump_task.done():
                pump_task.cancel()
        except Exception:
            pass
        try:
            if openai_ws and not openai_ws.closed:
                await openai_ws.close()
        except Exception:
            pass
        try:
            if openai_session:
                await openai_session.close()
        except Exception:
            pass

    async def send_turn_from_chunks(chunks: List[str]):
        """Send one response.create turn with inline input_audio (no buffer/commit)."""
        nonlocal pending
        if not chunks:
            return
        await send_openai({
            "type": "response.create",
            "input_audio": [{"audio": c, "format": "pcm16"} for c in chunks],
            "response": {
                "modalities": ["text", "audio"],
                "instructions": "Reply in English only. Keep it short."
            }
        })
        pending = True
        logger.info("turn sent: chunks=%d", len(chunks))

    try:
        while True:
            raw = await ws.receive_text()
            m = json.loads(raw)
            ev = m.get("event")

            if ev == "start":
                logger.info("Exotel stream started sr=8000")

            elif ev == "media":
                b64 = m.get("audio")
                if not b64:
                    media = m.get("media") or {}
                    b64 = media.get("payload")

                if not b64:
                    continue

                try:
                    blen = len(base64.b64decode(b64))
                except Exception:
                    blen = 0
                if blen == 0:
                    continue

                if not connected_to_openai:
                    await openai_connect()

                # If a response is pending or actively speaking, buffer into barge window.
                if pending or speaking:
                    barge_chunks.append(b64)
                    barge_bytes += blen
                    barge_frames += 1
                    if barge_bytes >= MIN_WINDOW and barge_frames >= 2:
                        logger.info("barge-in: cancel and send new turn (%d frames)", barge_frames)
                        await send_openai({"type": "response.cancel"})
                        pending = False
                        speaking = False
                        await asyncio.sleep(0)
                        await send_turn_from_chunks(barge_chunks)
                        barge_chunks.clear(); barge_bytes = barge_frames = 0
                    continue

                # Normal listening window
                live_chunks.append(b64)
                live_bytes += blen
                live_frames += 1

                if live_bytes >= MIN_WINDOW and live_frames >= 2 and not pending:
                    await send_turn_from_chunks(live_chunks)
                    live_chunks.clear(); live_bytes = live_frames = 0

            elif ev == "stop":
                logger.info("Exotel stream stopped")
                break

            else:
                # Unknown / ignored events
                pass

    except WebSocketDisconnect:
        logger.info("Exotel WS disconnected")
    except Exception as e:
        logger.exception("Exotel WS error: %s", e)
    finally:
        await openai_close()
        try:
            await ws.close()
        except Exception:
            pass


# --------------- Run locally ---------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ws_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
        reload=False,
        workers=1,
    )
