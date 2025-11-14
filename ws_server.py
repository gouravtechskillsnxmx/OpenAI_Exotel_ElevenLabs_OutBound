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


# ---------------- Realtime media bridge (Exotel <-> OpenAI) ----------------
@app.websocket("/exotel-media")
async def exotel_media_ws(ws: WebSocket):
    """
    Bidirectional audio bridge for Exotel Voicebot (callee leg @ 8 kHz PCM16).
    LIC insurance agent persona, bot greets first.
    """
    await ws.accept()
    logger.info("Exotel WS connected")

    if not OPENAI_API_KEY:
        logger.error("No OPENAI_API_KEY/OpenAI_Key configured; closing")
        await ws.close()
        return

    EXO_SR = 8000
    BYTES_PER_SAMPLE = 2
    MIN_WINDOW = int(EXO_SR * BYTES_PER_SAMPLE * 0.12)  # ~120ms

    pending = False
    speaking = False
    connected_to_openai = False
    intro_sent = False

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

        # LIC Agent persona
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
                "instructions": (
                    "You are an experienced Indian life insurance agent specializing in LIC-style "
                    "life insurance policies (term plans, endowment plans, etc.). "
                    "Speak in clear, friendly Indian English. "
                    "Your job on this call is to understand the customer's age, family responsibilities, "
                    "income range, and whether they prefer pure protection or some savings element. "
                    "Explain options at a high level without promising guaranteed returns, "
                    "without quoting exact premiums or specific policy numbers. "
                    "Always remind that final decisions must be taken with a licensed human insurance advisor "
                    "or at an LIC branch. Keep answers concise and conversational."
                )
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
        """Send one response.create turn with inline input_audio."""
        nonlocal pending
        if not chunks:
            return
        await send_openai({
            "type": "response.create",
            "input_audio": [{"audio": c, "format": "pcm16"} for c in chunks],
            "response": {
                "modalities": ["text", "audio"],
                "instructions": (
                    "Continue the conversation as the LIC-style insurance agent described earlier. "
                    "Ask focused questions about their needs, explain benefits simply, and stay brief."
                )
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

                if not connected_to_openai:
                    await openai_connect()

                if not intro_sent:
                    await send_openai({
                        "type": "response.create",
                        "response": {
                            "modalities": ["audio", "text"],
                            "instructions": (
                                "Start the call as a polite LIC-style life insurance agent. "
                                "Greet the customer by saying something like "
                                "'Namaste, I am your life insurance advisor calling from an LIC-style "
                                "insurance service.' "
                                "Briefly explain that you help people choose suitable life insurance "
                                "plans for family protection and future goals. "
                                "Then ask 1–2 simple questions: their age range, whether they have dependents, "
                                "and if they already have any life insurance. "
                                "Keep it short and friendly."
                            )
                        }
                    })
                    intro_sent = True
                    pending = True

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
