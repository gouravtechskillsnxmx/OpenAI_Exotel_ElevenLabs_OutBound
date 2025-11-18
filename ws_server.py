"""
ws_server.py — Exotel Outbound Realtime LIC Agent + Call Logs + CSV Dashboard
-----------------------------------------------------------------------------

Features:
- Outbound calls via Exotel Connect API to a Voicebot App/Flow (EXO_FLOW_ID)
- Realtime LIC insurance agent voicebot using OpenAI Realtime
- Exotel status webhook saving call details into SQLite
- Simple dashboard at /dashboard:
  - Upload CSV (number,name) to trigger outbound calls
  - View recent call logs

ENV (set in Render):
  EXO_SID           e.g. gouravnxmx1
  EXO_API_KEY       from Exotel API settings
  EXO_API_TOKEN     from Exotel API settings
  EXO_FLOW_ID       e.g. 1077390 (your Voicebot app id)
  EXO_SUBDOMAIN     api or api.in   (NOT the full domain)
  EXO_CALLER_ID     your Exophone, e.g. 09513886363

  OPENAI_API_KEY or OpenAI_Key or OPENAI_KEY
  OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview (optional)

  PUBLIC_BASE_URL   e.g. openai-exotel-elevenlabs-outbound.onrender.com
  LOG_LEVEL=INFO

  DB_PATH=/tmp/call_logs.db   (or /data/call_logs.db if you have persistent disk)
  SAVE_TTS_WAV=1              (optional: save bot audio WAVs in /tmp)
"""

import os, json, base64, asyncio, logging, time, wave, audioop, csv, sqlite3
from pathlib import Path
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
    Query,
)
from fastapi.responses import PlainTextResponse, JSONResponse, HTMLResponse
from aiohttp import ClientSession, WSMsgType
from pydantic import BaseModel
import numpy as np
from scipy.signal import resample


# ---------------- Logging ----------------
level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, level, logging.INFO))
logger = logging.getLogger("ws_server")

# ---------------- FastAPI ----------------
app = FastAPI(title="Exotel Outbound Realtime LIC Agent")


# ---------------- DB (SQLite) ----------------
DB_PATH = os.getenv("DB_PATH", "/tmp/call_logs.db")


def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS call_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_sid TEXT UNIQUE,
            direction TEXT,
            from_number TEXT,
            to_number TEXT,
            status TEXT,
            recording_url TEXT,
            started_at TEXT,
            ended_at TEXT,
            raw_payload TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    logger.info("SQLite DB initialized at %s", DB_PATH)


def upsert_call_log(data: dict):
    """
    Upsert (by CallSid) a record into call_logs.
    Exotel may send multiple status callbacks; we keep the latest.
    """
    call_sid = data.get("CallSid") or data.get("CallSid[]") or ""
    if not call_sid:
        return

    direction = data.get("Direction") or data.get("Direction[]") or ""
    frm = data.get("From") or data.get("From[]") or ""
    to = data.get("To") or data.get("To[]") or ""
    status = data.get("Status") or data.get("Status[]") or ""
    recording_url = (
        data.get("RecordingUrl")
        or data.get("RecordingUrl[]")
        or data.get("RecordingURL")
        or ""
    )
    started_at = data.get("StartTime") or data.get("StartTime[]") or ""
    ended_at = data.get("EndTime") or data.get("EndTime[]") or ""

    raw_payload = json.dumps(data, ensure_ascii=False)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO call_logs (
            call_sid, direction, from_number, to_number, status,
            recording_url, started_at, ended_at, raw_payload
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(call_sid) DO UPDATE SET
            direction=excluded.direction,
            from_number=excluded.from_number,
            to_number=excluded.to_number,
            status=excluded.status,
            recording_url=excluded.recording_url,
            started_at=excluded.started_at,
            ended_at=excluded.ended_at,
            raw_payload=excluded.raw_payload
        """,
        (
            call_sid,
            direction,
            frm,
            to,
            status,
            recording_url,
            started_at,
            ended_at,
            raw_payload,
        ),
    )
    conn.commit()
    conn.close()
    logger.info(
        "call_log upserted: sid=%s status=%s from=%s to=%s recording=%s",
        call_sid,
        status,
        frm,
        to,
        recording_url,
    )


init_db()


# ---------------- Request models ----------------
class OutboundCallRequest(BaseModel):
    """Body for /exotel-outbound-call"""
    to_number: str   # customer mobile/landline, e.g. "8850298070"


# ---------------- Exotel ENV ----------------
EXO_SID       = os.getenv("EXO_SID", "")
EXO_API_KEY   = os.getenv("EXO_API_KEY", "")
EXO_API_TOKEN = os.getenv("EXO_API_TOKEN", "")
EXO_FLOW_ID   = os.getenv("EXO_FLOW_ID", "")            # App / Flow app id (e.g. 1077390)
EXO_SUBDOMAIN = os.getenv("EXO_SUBDOMAIN", "api")       # "api" or "api.in"
EXO_CALLER_ID = os.getenv("EXO_CALLER_ID", "")          # Your Exophone


# ---------------- OpenAI ENV ----------------
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OpenAI_Key")
    or os.getenv("OPENAI_KEY", "")
)
REALTIME_MODEL = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")

# ---------------- Misc ENV ----------------
PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").strip()  # no protocol
SAVE_TTS_WAV    = os.getenv("SAVE_TTS_WAV", "0") == "1"


# ---------------- Helper: downsample ----------------
def downsample_24k_to_8k_pcm16(pcm24: bytes) -> bytes:
    """24 kHz mono PCM16 -> 8 kHz mono PCM16 using stdlib audioop."""
    converted, _ = audioop.ratecv(pcm24, 2, 1, 24000, 8000, None)
    return converted

def upsample_8k_to_24k_pcm16(pcm8: bytes) -> bytes:
    """8 kHz mono PCM16 -> 24 kHz mono PCM16 using stdlib audioop."""
    converted, _ = audioop.ratecv(pcm8, 2, 1, 8000, 24000, None)
    return converted



# ---------------- Helper: Exotel outbound (Connect API) ----------------
async def exotel_connect_voicebot(to_e164: str) -> dict:
    """
    Start an outbound call via Exotel Connect API and drop the callee
    into your Voicebot App/Flow (EXO_FLOW_ID) which points to /exotel-ws-bootstrap.
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

    base = f"https://{EXO_SUBDOMAIN}.exotel.com"
    url = f"{base}/v1/Accounts/{EXO_SID}/Calls/connect.json"

    exoml_url = f"https://my.exotel.com/{EXO_SID}/exoml/start_voice/{EXO_FLOW_ID}"

    payload = {
        "From": to_e164,
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
    realtime LIC insurance agent via the Exotel Voicebot App (EXO_FLOW_ID).
    """
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
            await asyncio.sleep(0.5)
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
    Configure in Exotel Voicebot / App to POST here.
    """
    try:
        form = await request.form()
        data = dict(form)
    except Exception:
        data = {}

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

    upsert_call_log(data)
    return JSONResponse({"ok": True})


# ---------------- API to fetch call logs ----------------
@app.get("/calls")
async def list_calls(limit: int = Query(50, ge=1, le=500)):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        SELECT call_sid, direction, from_number, to_number, status,
               recording_url, started_at, ended_at
        FROM call_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = c.fetchall()
    conn.close()

    result = []
    for r in rows:
        result.append(
            {
                "call_sid": r[0],
                "direction": r[1],
                "from_number": r[2],
                "to_number": r[3],
                "status": r[4],
                "recording_url": r[5],
                "started_at": r[6],
                "ended_at": r[7],
            }
        )
    return {"calls": result}


# ---------------- Bootstrap for Exotel Voicebot ----------------
@app.get("/exotel-ws-bootstrap")
async def exotel_ws_bootstrap():
    try:
        base = PUBLIC_BASE_URL or "openai-exotel-elevenlabs-outbound.onrender.com"
        url = f"wss://{base}/exotel-media"
        logger.info("Bootstrap served: %s", url)
        return {"url": url}
    except Exception as e:
        logger.exception("/exotel-ws-bootstrap error: %s", e)
        return {"url": f"wss://{(PUBLIC_BASE_URL or 'openai-exotel-elevenlabs-outbound.onrender.com')}/exotel-media"}


@app.websocket("/exotel-media")
async def exotel_media_ws(ws: WebSocket):
    await ws.accept()
    logger.info("Exotel WS connected (Shashinath LIC agent, realtime)")

    if not OPENAI_API_KEY:
        logger.error("No OPENAI_API_KEY; closing Exotel stream.")
        await ws.close()
        return

    # --------- Stream state ---------
    stream_sid: Optional[str] = None

    # Incoming audio from Exotel
    sample_rate: int = 8000       # Exotel media sample rate (updated from "start")
    target_sr: int = 24000        # OpenAI realtime expects 24k PCM16
    bytes_per_sample: int = 2     # PCM16 mono

    # We will only commit when we know we have enough audio
    min_commit_ms: float = 120.0  # >=100ms required by Realtime; we use 120ms for safety
    silence_duration_ms: float = 600.0  # end-of-turn if no audio for 600ms

    buffered_ms: float = 0.0      # how much caller audio we’ve sent since last commit
    last_audio_time: float = 0.0  # last time we got caller audio
    silence_check_task: Optional[asyncio.Task] = None

    # Outgoing audio to Exotel
    seq_num = 1       # Exotel sequence_number
    chunk_num = 1     # Exotel media.chunk
    start_ts = time.time()

    # Realtime/OpenAI
    openai_session: Optional[ClientSession] = None
    openai_ws = None
    openai_reader_task: Optional[asyncio.Task] = None
    speaking: bool = False        # True while bot is sending audio
										 
											 
																 
				  
							   
										
										  

    async def send_audio_to_exotel(pcm8: bytes):
        """Send 8k PCM16 audio back to Exotel as proper media frames."""
        nonlocal seq_num, chunk_num, start_ts, stream_sid

        if not stream_sid:
            logger.warning("No stream_sid; cannot send audio to Exotel yet")
            return

        FRAME_BYTES = 320  # 20 ms at 8kHz mono 16-bit
        now_ms = lambda: int((time.time() - start_ts) * 1000)

        for i in range(0, len(pcm8), FRAME_BYTES):
            chunk_bytes = pcm8[i:i + FRAME_BYTES]
            if not chunk_bytes:
                continue

            payload_b64 = base64.b64encode(chunk_bytes).decode("ascii")
            ts = now_ms()

            msg = {
                "event": "media",
                "stream_sid": stream_sid,
                "sequence_number": str(seq_num),
                "media": {
                    "chunk": str(chunk_num),
                    "timestamp": str(ts),
                    "payload": payload_b64,
                },
            }

            await ws.send_text(json.dumps(msg))
            logger.info(
                "Sent audio media to Exotel (seq=%s, chunk=%s, bytes=%s)",
                seq_num, chunk_num, len(chunk_bytes),
            )

            seq_num += 1
            chunk_num += 1

    async def openai_connect():
        """Open the Realtime WS to OpenAI and configure the Shashinath LIC persona."""
        nonlocal openai_session, openai_ws, openai_reader_task

			
					   
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
                   "OpenAI-Beta": "realtime=v1"}
			 
        url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

        openai_session = ClientSession()
															  
        openai_ws = await openai_session.ws_connect(url, headers=headers)
        logger.info("Connected to OpenAI Realtime WS")

        # Session config: PCM16 in/out, manual commit based on silence, LIC persona
        await openai_ws.send_json({
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": None,   # WE manage turns with silence + commit
									   
                "voice": "alloy",
										 
												  
					  
                "instructions": (
                    "You are Mr. Shashinath Thakur, a senior LIC life insurance advisor "
                    "based in Mumbai. You speak in friendly Hinglish (mix of Hindi and English), "
                    "calm and trustworthy, like a real LIC agent on a phone call. "
                    "Help callers with LIC life insurance, term plans, premiums, riders, "
                    "maturity values, tax benefits, and claim process. "
                    "In your FIRST reply, clearly introduce yourself as 'LIC agent "
                    "Mr. Shashinath Thakur from Mumbai'. "
                    "For every reply, speak very briefly: about 1–2 sentences, 8–12 words, "
                    "then stop and wait silently for the caller. "
                    "Never talk about topics outside LIC insurance and basic financial planning."
                ),
            }
        })

        async def pump_openai_to_exotel():
            nonlocal speaking
            tts_dump: bytearray = bytearray()
							 
													
									 
														   
																					  
																					   
																								 
																									
													  
																			  
					  
				  
			  

							 
            try:
                async for msg in openai_ws:
                    if msg.type == WSMsgType.TEXT:
									
                        evt = msg.json()
                        etype = evt.get("type")
                        logger.info("OpenAI EVENT: %s", etype)

                        if etype == "response.audio.delta":
																	 
                            chunk_b64 = evt.get("delta")
                            if chunk_b64 and ws.client_state.name != "DISCONNECTED":
                                pcm24 = base64.b64decode(chunk_b64)
                                if SAVE_TTS_WAV:
                                    tts_dump.extend(pcm24)

														 
                                pcm8 = downsample_24k_to_8k_pcm16(pcm24)
                                speaking = True
                                await send_audio_to_exotel(pcm8)

                        elif etype in ("response.audio.done", "response.completed", "response.done"):
                            logger.info("OpenAI finished a response turn")
                            speaking = False

                        elif etype == "error":
																								
                            logger.error("OpenAI error event: %s", evt)
                            break

                    elif msg.type == WSMsgType.ERROR:
                        logger.error("OpenAI WS error")
                        break
            except Exception as e:
                logger.exception("OpenAI pump error: %s", e)
            finally:
                if SAVE_TTS_WAV and tts_dump:
                    fname = f"/tmp/openai_tts_{int(time.time())}.wav"
                    with wave.open(fname, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        wf.writeframes(bytes(tts_dump))
                    logger.info("Saved OpenAI TTS to %s", fname)

        openai_reader_task = asyncio.create_task(pump_openai_to_exotel())
														 

    async def openai_close():
        """Gracefully close OpenAI WS and session."""
        nonlocal silence_check_task
        if silence_check_task:
            silence_check_task.cancel()
        try:
            if openai_reader_task and not openai_reader_task.done():
                openai_reader_task.cancel()
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

    # Silence checker: detect end-of-turn and trigger commit + response.create
    async def silence_checker():
        nonlocal buffered_ms, last_audio_time
        loop = asyncio.get_event_loop()
        while True:
            await asyncio.sleep(0.1)
            if last_audio_time == 0.0:
                continue
            now = loop.time()
            if (now - last_audio_time) * 1000.0 > silence_duration_ms:
                if buffered_ms >= min_commit_ms:
                    logger.info(
                        "Silence %.0fms and buffered_ms=%.1fms -> commit & respond",
                        (now - last_audio_time) * 1000.0,
                        buffered_ms,
                    )
                    # Tell Realtime to treat what we've sent as one user turn
                    await openai_ws.send_json({"type": "input_audio_buffer.commit"})
                    await openai_ws.send_json({
                        "type": "response.create",
                        "response": {
                            "modalities": ["audio", "text"],
                            "instructions": (
                                "Reply as LIC agent Mr. Shashinath Thakur in friendly Hinglish. "
                                "Give a very short answer (1–2 sentences), then stop."
                            ),
                        },
                    })
                    buffered_ms = 0.0
                    last_audio_time = loop.time()

    # Connect to OpenAI and start silence checker
    await openai_connect()
    silence_check_task = asyncio.create_task(silence_checker())

    loop = asyncio.get_event_loop()

    try:
        while True:
            raw = await ws.receive_text()
            evt = json.loads(raw)
            etype = evt.get("event")
            logger.info("Exotel EVENT: %s - msg=%s", etype, evt)

            if etype == "connected":
								
                continue

            elif etype == "start":
                start_obj = evt.get("start", {})
                stream_sid = start_obj.get("stream_sid") or start_obj.get("streamSid")
                mf = start_obj.get("media_format") or {}
                sample_rate = int(mf.get("sample_rate") or sample_rate)
                logger.info("Exotel stream started sid=%s sr=%d", stream_sid, sample_rate)
                start_ts = time.time()
                last_audio_time = loop.time()
												

            elif etype == "media":
																								  
                media = evt.get("media") or {}
                payload_b64 = media.get("payload")
																	  
                if not payload_b64:
                    continue
																	   

                if openai_ws is None or openai_ws.closed:
                    logger.warning("OpenAI WS not ready; skipping audio frame")
                    continue
					  
																							 
																			 

                # Decode input audio from Exotel
                try:
                    audio_bytes = base64.b64decode(payload_b64)
                    if len(audio_bytes) == 0:
                        continue
                except Exception:
                    logger.warning("Invalid base64 in media payload")
                    continue

                # Approximate ms in this frame and accumulate
                samples = len(audio_bytes) / bytes_per_sample
                frame_ms = (samples / sample_rate) * 1000.0
                buffered_ms += frame_ms

                # HARD BARGE-IN: if bot is speaking and user talks, cancel current response
                if speaking:
                    logger.info("Barge-in: caller spoke while bot speaking, cancelling response")
                    try:
                        await openai_ws.send_json({"type": "response.cancel"})
                    except Exception as e:
                        logger.exception("Error sending response.cancel: %s", e)
                    speaking = False

                # Resample caller audio from sample_rate -> 24kHz
                if sample_rate != target_sr:
                    try:
                        samples_arr = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                        resample_ratio = target_sr / sample_rate
                        target_samples = int(len(samples_arr) * resample_ratio)
                        if target_samples <= 0:
                            continue
                        resampled = resample(samples_arr, target_samples)
                        resampled = np.clip(resampled, -32768, 32767).astype(np.int16)
                        resampled_bytes = resampled.tobytes()
                        resampled_b64 = base64.b64encode(resampled_bytes).decode("utf-8")
                    except Exception as e:
                        logger.error("Resample failed: %s", e)
                        continue
                else:
                    resampled_b64 = payload_b64

                # Append to OpenAI buffer (we'll commit on silence)
                await openai_ws.send_json({
                    "type": "input_audio_buffer.append",
                    "audio": resampled_b64
                })
                last_audio_time = loop.time()

            elif etype == "dtmf":
                # ignore for now
                pass

            elif etype == "stop":
                logger.info("Exotel stream stopped sid=%s", stream_sid)
                break

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

# ---------------- Simple CSV + Logs Dashboard ----------------
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>LIC Outbound Voicebot Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    h1 { margin-bottom: 0; }
    .box { border: 1px solid #ccc; padding: 15px; margin: 15px 0; border-radius: 6px; }
    input, button { padding: 8px; margin: 4px 0; }
    #log { white-space: pre-line; border: 1px solid #ddd; padding: 10px; height: 200px; overflow-y: scroll; }
    table { border-collapse: collapse; width: 100%; margin-top: 10px; }
    th, td { border: 1px solid #ddd; padding: 6px; font-size: 13px; }
    th { background: #f0f0f0; }
  </style>
</head>
<body>
  <h1>LIC Outbound Voicebot Dashboard</h1>
  <p>Backend: Exotel Connect + OpenAI Realtime (LIC insurance agent persona)</p>

  <div class="box">
    <h2>Single Call Test</h2>
    <input id="single-number" placeholder="Mobile number (10 digits)"><br>
    <button onclick="singleCall()">Call Now</button>
  </div>

  <div class="box">
    <h2>CSV Campaign</h2>
    <p>Upload CSV with columns: <code>number,name</code></p>
    <input type="file" id="csv-file">
    <button onclick="uploadCSV()">Upload & Call</button>
  </div>

  <div class="box">
    <h2>Recent Call Logs</h2>
    <button onclick="loadCalls()">Refresh</button>
    <table id="calls-table">
      <thead>
        <tr>
          <th>CallSid</th>
          <th>From</th>
          <th>To</th>
          <th>Status</th>
          <th>Start</th>
          <th>End</th>
          <th>Recording</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>

  <div class="box">
    <h2>Log</h2>
    <div id="log"></div>
  </div>

<script>
const BASE = window.location.origin;

function log(msg) {
  const el = document.getElementById("log");
  el.innerText += msg + "\\n";
  el.scrollTop = el.scrollHeight;
}

async function singleCall() {
  const num = document.getElementById("single-number").value.trim();
  if (!num) { alert("Enter a number"); return; }

  const payload = { to_number: num };
  log("Calling " + num + " ...");
  const res = await fetch(BASE + "/exotel-outbound-call", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  log("Response: " + JSON.stringify(data));
}

async function uploadCSV() {
  const fileInput = document.getElementById("csv-file");
  if (!fileInput.files.length) { alert("Choose a CSV file first"); return; }
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  log("Uploading CSV and triggering calls ...");
  const res = await fetch(BASE + "/outbound/csv", {
    method: "POST",
    body: formData
  });
  const data = await res.json();
  log("CSV result: " + JSON.stringify(data));
}

async function loadCalls() {
  const res = await fetch(BASE + "/calls?limit=100");
  const data = await res.json();
  const tbody = document.querySelector("#calls-table tbody");
  tbody.innerHTML = "";
  (data.calls || []).forEach(c => {
    const tr = document.createElement("tr");
    const recLink = c.recording_url
      ? '<a href="' + c.recording_url + '" target="_blank">Play</a>'
      : '';
    tr.innerHTML =
      "<td>" + (c.call_sid || "") + "</td>" +
      "<td>" + (c.from_number || "") + "</td>" +
      "<td>" + (c.to_number || "") + "</td>" +
      "<td>" + (c.status || "") + "</td>" +
      "<td>" + (c.started_at || "") + "</td>" +
      "<td>" + (c.ended_at || "") + "</td>" +
      "<td>" + recLink + "</td>";
    tbody.appendChild(tr);
  });
  log("Loaded " + (data.calls || []).length + " calls");
}

loadCalls();
</script>
</body>
</html>
    """


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
