"""
Voice Agent with Integrated Routing System
Main entry point for the voice agent with navigation, medical extraction, and general tools
"""

import os
import asyncio
import traceback
from dotenv import load_dotenv

load_dotenv()

import pyaudio

from google import genai
from google.genai import types

# Import handlers
from handlers.navigation import NavigationHandler
from handlers.medical_extraction import MedicalExtractionHandler  
from handlers.general_tools import GeneralToolsHandler

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"

client = genai.Client(
    http_options={"api_version": "v1beta"},
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# Initialize handlers
navigation_handler = NavigationHandler()
medical_handler = MedicalExtractionHandler()
general_handler = GeneralToolsHandler()

CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    system_instruction="""You are a helpful voice assistant with three main capabilities:

1. **Navigation**: When users ask to go to pages or websites, use the navigate_to_page tool.
   Available pages: home (Google), gemini (Gemini), profile, settings, and dashboard.
   Examples: 'go to Google', 'open Gemini', 'take me to settings'

2. **Medical Information Extraction**: When users describe medical procedures, biopsies, or anatomical sites, use the extract_medical_info tool.
   Examples: 'biopsy from antrum', 'polyp in sigmoid colon', 'sample from oesophagus'

3. **General Tools**: Use appropriate tools for:
   - Weather queries: get_weather_tool
   - Cryptocurrency prices: crypto_api  
   - News: news_api
   - Jokes: joke_api
   - Quotes: quote_api

Listen carefully to determine which tool to use based on the user's intent. And answer only in **english**""",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        ),
        language_code="en-US"
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=4000,
        sliding_window=types.SlidingWindow(target_tokens=2000),
    ),
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
    tools=[
        navigation_handler.get_tool(),
        medical_handler.get_tool(),
        general_handler.get_tools(),
    ],
)

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self):
        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        self.last_speaker = None
        self.ai_speaking = False

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            if not self.ai_speaking:
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        """Background task to reads from the websocket and write pcm chunks to the output queue"""
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                if (
                    server_content := response.server_content
                ) and server_content.input_transcription:
                    if self.last_speaker != "User":
                        print("\\nUser: ", end="")
                        self.last_speaker = "User"
                    print(server_content.input_transcription.text, end="", flush=True)
                if (
                    server_content := response.server_content
                ) and server_content.output_transcription:
                    if self.last_speaker != "Model":
                        print("\\nModel: ", end="")
                        self.last_speaker = "Model"
                    print(server_content.output_transcription.text, end="", flush=True)

                if tool_call := response.tool_call:
                    for fc in tool_call.function_calls:
                        result = None
                        
                        # Route to appropriate handler based on tool name
                        if fc.name == "navigate_to_page":
                            result = navigation_handler.handle(**fc.args)
                            
                        elif fc.name == "extract_medical_info":
                            result = medical_handler.handle(**fc.args)
                            print(f"\\n[MEDICAL] Extracted info: {result}")
                        
                        # Handle general tools (weather, news, crypto, etc.)
                        else:
                            result = general_handler.handle(fc.name, **fc.args)
                            print(f"\\n[GENERAL_TOOLS] {fc.name} result: {result}")
                        
                        if result:
                            await self.session.send(
                                input=types.LiveClientToolResponse(
                                    function_responses=[
                                        types.FunctionResponse(
                                            name=fc.name,
                                            id=fc.id,
                                            response=result,
                                        )
                                    ]
                                )
                            )

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            self.ai_speaking = True
            await asyncio.to_thread(stream.write, bytestream)
            self.ai_speaking = False

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


if __name__ == "__main__":
    main = AudioLoop()
    asyncio.run(main.run())
