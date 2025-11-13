import os, asyncio, json
from aiohttp import ClientSession, WSMsgType

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

async def main():
    assert API_KEY, "No OpenAI key in env"
    headers = {"Authorization": f"Bearer {API_KEY}", "OpenAI-Beta": "realtime=v1"}
    async with ClientSession() as s:
        async with s.ws_connect(URL, headers=headers) as ws:
            # Configure session to μ-law 8 kHz for Twilio compatibility
            await ws.send_json({
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
                    "instructions": "You are a concise helpful voice agent. Keep answers short."
                }
            })
            # Ask for audio from a short text prompt
            await ws.send_json({
                "type": "response.create",
                "response": {
                    "modalities": ["text","audio"],
                    "instructions": "Say 'realtime check successful' briefly."
                }
            })
            while True:
                msg = await ws.receive()
                if msg.type == WSMsgType.TEXT:
                    evt = msg.json()
                    t = evt.get("type")
                    if t == "response.audio.delta":
                        if evt.get("delta"):
                            print("got audio chunk:", len(evt["delta"]))
                    elif t == "response.completed":
                        print("completed")
                        break
                    elif t == "error":
                        print("ERROR:", evt); break
                elif msg.type in (WSMsgType.CLOSED, WSMsgType.ERROR):
                    print("socket closed"); break

asyncio.run(main())
