import os
import asyncio
from dotenv import load_dotenv

load_dotenv()
import base64
import io
import traceback
import wave
from datetime import datetime

import pyaudio
from pydub import AudioSegment

from google import genai
from google.genai import types

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


CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "AUDIO",
    ],
    system_instruction="You must ALWAYS respond in English only, regardless of what language the user speaks in.",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Alnilam")
        ),
        # language_code="en-US"
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
)

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, audio_file_path=None):
        self.audio_in_queue = None
        self.session = None
        self.audio_file_path = audio_file_path
        self.last_speaker = None
        self.ai_speaking = False
        self.file_sent = False
        
        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = f"response_{timestamp}.wav"
        self.wav_file = None

    async def send_audio_file(self):
        """Send pre-recorded audio file to the API (converts to PCM format)"""
        if not self.audio_file_path:
            print("No audio file provided. Please provide a file path.")
            return
        
        if not os.path.exists(self.audio_file_path):
            print(f"Audio file not found: {self.audio_file_path}")
            return
        
        print(f"Processing audio file: {self.audio_file_path}")
        
        try:
            # Load audio file and convert to PCM format
            audio = AudioSegment.from_file(self.audio_file_path)
            
            # Convert to the format expected by Gemini Live API
            audio = audio.set_frame_rate(SEND_SAMPLE_RATE)  # 16kHz
            audio = audio.set_channels(CHANNELS)  # Mono
            audio = audio.set_sample_width(2)  # 16-bit
            
            # Export to raw PCM data
            pcm_data = audio.raw_data
            
            print(f"Converted to PCM: {len(pcm_data)} bytes, {len(pcm_data) / (SEND_SAMPLE_RATE * 2):.2f} seconds")
            
            # Send PCM data in chunks (simulate streaming)
            chunk_size = CHUNK_SIZE * 2  # 2 bytes per sample for 16-bit audio
            
            for i in range(0, len(pcm_data), chunk_size):
                chunk = pcm_data[i:i + chunk_size]
                # Use the correct API: audio parameter expects a Blob
                await self.session.send_realtime_input(
                    audio={"data": chunk, "mime_type": "audio/pcm"}
                )
                await asyncio.sleep(0.01)  # Small delay to simulate real-time streaming
            
            # Signal end of audio stream
            await self.session.send_realtime_input(audio_stream_end=True)
            
            self.file_sent = True
            print(f"Audio file sent successfully")
            
        except Exception as e:
            print(f"Error processing audio file: {e}")
            import traceback
            traceback.print_exc()
            self.file_sent = True  # Allow the program to continue

    async def send_text(self):
        """Allow user to send additional text queries after processing the audio file"""
        # Wait for the audio file to be sent first
        while not self.file_sent:
            await asyncio.sleep(0.1)
        
        # Optional: wait a bit for the response to complete
        await asyncio.sleep(2)
        
        while True:
            text = await asyncio.to_thread(
                input,
                "\nmessage > ",
            )
            if text.lower() == "q":
                break
            # Use text parameter for send_realtime_input
            await self.session.send_realtime_input(text=text or ".")

    async def receive_audio(self):
        """Background task to read from the websocket and write pcm chunks to the output queue"""
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                if (
                    server_content := response.server_content
                ) and server_content.input_transcription:
                    if self.last_speaker != "User":
                        print("\nUser: ", end="")
                        self.last_speaker = "User"
                    print(server_content.input_transcription.text, end="", flush=True)
                if (
                    server_content := response.server_content
                ) and server_content.output_transcription:
                    if self.last_speaker != "Model":
                        print("\nModel: ", end="")
                        self.last_speaker = "Model"
                    print(server_content.output_transcription.text, end="", flush=True)

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        """Play the audio response from the API and save to WAV file"""
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        
        # Open WAV file for writing
        self.wav_file = wave.open(self.output_file, 'wb')
        self.wav_file.setnchannels(CHANNELS)
        self.wav_file.setsampwidth(pya.get_sample_size(FORMAT))
        self.wav_file.setframerate(RECEIVE_SAMPLE_RATE)
        
        print(f"Saving audio response to: {self.output_file}")
        
        while True:
            bytestream = await self.audio_in_queue.get()
            self.ai_speaking = True
            
            # Play audio through speakers
            await asyncio.to_thread(stream.write, bytestream)
            
            # Write audio to WAV file
            await asyncio.to_thread(self.wav_file.writeframes, bytestream)
            
            self.ai_speaking = False

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.audio_in_queue = asyncio.Queue()

                # Send the audio file first
                tg.create_task(self.send_audio_file())
                # Handle text input for follow-up queries
                send_text_task = tg.create_task(self.send_text())
                # Receive and play audio responses
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            traceback.print_exception(EG)
        finally:
            # Close WAV file if it was opened
            if self.wav_file:
                self.wav_file.close()
                print(f"\n\nAudio response saved to: {os.path.abspath(self.output_file)}")


if __name__ == "__main__":
    import sys
    
    # Get audio file path from command line argument or prompt user
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = input("Enter the path to your audio file: ").strip()
    
    main = AudioLoop(audio_file_path=audio_file)
    asyncio.run(main.run())