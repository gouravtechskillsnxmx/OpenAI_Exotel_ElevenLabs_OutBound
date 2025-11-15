# ws_server.py — Exotel Outbound Realtime LIC Agent + Call Logs + CSV Dashboard
# -----------------------------------------------------------------------------

# ... (all imports and earlier code unchanged) ...

@app.websocket("/exotel-media")
async def exotel_media_ws(ws: WebSocket):
    """
    Bidirectional audio bridge for Exotel Voicebot (callee leg @ 8 kHz PCM16).
    LIC insurance agent persona, bot greets first.
    """
    await ws.accept()
    logger.info("Exotel WS connected")

    if not OPENAI_API_KEY:
        logger.error("No OPENAI_API_KEY; closing Exotel stream.")
        await ws.close()
        return

    # Stream state
    stream_sid: Optional[str] = None
    sample_rate: int = 8000  # default; updated from "start"
    target_sr: int = 24000  # OpenAI required
    bytes_per_sample: int = 2  # PCM16 mono
    silence_duration_ms: float = 600  # Match session silence_duration_ms

    openai_session: Optional[ClientSession] = None
    openai_ws = None
    pump_task: Optional[asyncio.Task] = None
    connected_to_openai = False

    intro_sent = False
    speaking = False
    pending = False

    # Accumulators for user turn (manual mode not used; server VAD handles)
    live_chunks: list[str] = []
    live_bytes = 0
    live_frames = 0

    # Accumulators while bot is speaking (barge-in)
    barge_chunks: list[str] = []
    barge_bytes = 0
    barge_frames = 0

    MIN_WINDOW = int(8000 * bytes_per_sample * 0.15)  # ~150ms safe min for barge

    async def send_openai(payload: dict):
        if openai_ws is None or openai_ws.closed:
            logger.info("drop %s: OpenAI ws not ready/closed", payload.get("type"))
            return
        t = payload.get("type")
        if t != "response.audio.delta":
            logger.info("SENDING to OpenAI: %s", t)

        await openai_ws.send_json(payload)

    async def openai_connect():
        nonlocal openai_session, openai_ws, pump_task, connected_to_openai, speaking, pending
        if connected_to_openai:
            return

        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "OpenAI-Beta": "realtime=v1"}
        url = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

        openai_session = ClientSession()
        openai_ws = await openai_session.ws_connect(url, headers=headers)

        await send_openai({
            "type": "session.update",
            "session": {
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": silence_duration_ms
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
        """
        Append audio chunks to OpenAI's input_audio_buffer, commit, then
        request a LIC-style response.
        """
        nonlocal pending
        if not chunks:
            return

        # 1) Append each chunk to the server-side input_audio_buffer
        for c in chunks:
            await send_openai({
                "type": "input_audio_buffer.append",
                "audio": c,
            })

        # 2) Commit the buffer (now it has audio, so no "buffer too small" error)
        await send_openai({
            "type": "input_audio_buffer.commit",
        })

        # 3) Ask the model to respond (no input_audio parameter here)
        await send_openai({
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "instructions": (
                    "Continue the conversation as the LIC-style life insurance agent described earlier. "
                    "Ask focused questions about their needs, explain benefits simply, and keep responses concise."
                ),
            },
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

# ... (rest of the code unchanged) ...