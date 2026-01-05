import os
import asyncio
from dotenv import load_dotenv

load_dotenv()
import base64
import io
import traceback

import pyaudio

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
    # speech_config=types.SpeechConfig(
    #     voice_config=types.VoiceConfig(
    #         prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Alnilam")
    #     ),
    #     # language_code="en-US"
    # ),
    # context_window_compression=types.ContextWindowCompressionConfig(
    #     trigger_tokens=25600,
    #     sliding_window=types.SlidingWindow(target_tokens=12800),
    # ),
    # thinking_config=types.ThinkingConfig(
    #     thinking_budget=2000
    #     # thinking_level= 'low',
    # ),
        
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
)

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self):
        self.audio_in_queue = None
        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        self.play_audio_task = None
        self.last_speaker = None
        self.ai_speaking = False
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_thinking_tokens = 0
        self.turn_count = 0
        
        # Enhanced tracking for breakdown
        self.first_turn_input_tokens = 0  # Base API overhead
        self.previous_turn_input_tokens = 0  # For delta calculation
        self.base_overhead_calculated = False

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

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
                await self.session.send(input={"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            turn_input_tokens = 0
            turn_output_tokens = 0
            turn_thinking_tokens = 0
            
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
                
                # Track token usage - Comprehensive tracking
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    
                    # Core token counts
                    if hasattr(usage, 'prompt_token_count') and usage.prompt_token_count:
                        turn_input_tokens = usage.prompt_token_count
                    if hasattr(usage, 'response_token_count') and usage.response_token_count:
                        turn_output_tokens = usage.response_token_count
                    if hasattr(usage, 'thoughts_token_count') and usage.thoughts_token_count:
                        turn_thinking_tokens = usage.thoughts_token_count

            # Update total counters after turn completes
            if turn_input_tokens > 0 or turn_output_tokens > 0:
                self.total_input_tokens += turn_input_tokens
                self.total_output_tokens += turn_output_tokens
                self.total_thinking_tokens += turn_thinking_tokens
                self.turn_count += 1
                
                # Track first turn for base overhead calculation
                if not self.base_overhead_calculated:
                    self.first_turn_input_tokens = turn_input_tokens
                    self.base_overhead_calculated = True
                
                # Get the last usage metadata for detailed display
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    
                    # Extract modality details
                    current_audio_tokens = 0
                    current_text_tokens = 0
                    output_audio_tokens = 0
                    output_text_tokens = 0
                    
                    # Input modality breakdown
                    if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                        for detail in usage.prompt_tokens_details:
                            modality = detail.modality.name if hasattr(detail.modality, 'name') else str(detail.modality)
                            if modality == 'AUDIO':
                                current_audio_tokens = detail.token_count
                            elif modality == 'TEXT':
                                current_text_tokens = detail.token_count
                    
                    # Output modality breakdown
                    if hasattr(usage, 'response_tokens_details') and usage.response_tokens_details:
                        for detail in usage.response_tokens_details:
                            modality = detail.modality.name if hasattr(detail.modality, 'name') else str(detail.modality)
                            if modality == 'AUDIO':
                                output_audio_tokens = detail.token_count
                            elif modality == 'TEXT':
                                output_text_tokens = detail.token_count
                    
                    # Calculate input breakdown
                    base_overhead = 0
                    history_tokens = 0
                    current_message_tokens = current_audio_tokens
                    
                    if self.turn_count == 1:
                        base_overhead = current_text_tokens
                    else:
                        # For subsequent turns
                        token_delta = turn_input_tokens - self.previous_turn_input_tokens
                        history_tokens = token_delta - current_audio_tokens
                        base_overhead = current_text_tokens - history_tokens
                    
                    # Calculate output breakdown (estimate text content vs audio encoding)
                    # Rough estimate: ~1 token per word for text, rest is audio encoding
                    estimated_text_tokens = max(2, turn_output_tokens // 8)  # Rough estimate
                    estimated_audio_encoding = turn_output_tokens - estimated_text_tokens
                    
                    # Print comprehensive token usage for this turn
                    print(f"\n\n{'='*70}")
                    print(f"🔷 TURN #{self.turn_count} - DETAILED TOKEN BREAKDOWN")
                    print(f"{'='*70}\n")
                    
                    # ===== INPUT TOKENS SECTION =====
                    print(f"📥 INPUT TOKENS: {turn_input_tokens} total")
                    print(f"{'─'*70}")
                    print(f"   ┌─ Base API Overhead:     {base_overhead:4d} tokens")
                    print(f"   │  ├─ System instruction (~20-30 tokens)")
                    print(f"   │  ├─ Model configuration (~10-20 tokens)")
                    print(f"   │  └─ API setup/protocol (~{base_overhead - 50} tokens)")
                    
                    if self.turn_count > 1:
                        print(f"   ├─ Conversation History:  {history_tokens:4d} tokens")
                        print(f"   │  └─ Previous {self.turn_count - 1} turn(s) converted to text")
                    
                    print(f"   └─ Your Current Message:  {current_message_tokens:4d} tokens (AUDIO)")
                    
                    if self.turn_count > 1:
                        token_delta = turn_input_tokens - self.previous_turn_input_tokens
                        print(f"\n   📈 Token Growth: +{token_delta} tokens from Turn {self.turn_count - 1}")
                        print(f"      (History grew by ~{history_tokens} tokens)")
                    
                    # ===== OUTPUT TOKENS SECTION =====
                    print(f"\n📤 OUTPUT TOKENS: {turn_output_tokens} total")
                    print(f"{'─'*70}")
                    print(f"   ┌─ Estimated Text Content:   ~{estimated_text_tokens:3d} tokens")
                    print(f"   │  └─ Actual words/meaning of response")
                    print(f"   └─ Audio Encoding:         ~{estimated_audio_encoding:3d} tokens")
                    print(f"      ├─ Voice prosody (tone, pitch, rhythm)")
                    print(f"      ├─ Speech timing and pauses")
                    print(f"      └─ Audio waveform data")
                    
            
                    # ===== THINKING TOKENS SECTION =====
                    if turn_thinking_tokens > 0:
                        print(f"\n🧠 THINKING TOKENS: {turn_thinking_tokens} total")
                        print(f"{'─'*70}")
                        print(f"   └─ Internal reasoning (not shown to you)")
                        print(f"      └─ Model's thought process before responding")
                    
                    # ===== TOTAL SECTION =====
                    total_turn = turn_input_tokens + turn_output_tokens + turn_thinking_tokens
                    print(f"\n📊 TURN #{self.turn_count} TOTAL: {total_turn} tokens")
                    print(f"{'─'*70}")
                    print(f"   Input:    {turn_input_tokens:5d} tokens ({turn_input_tokens/total_turn*100:.1f}%)")
                    print(f"   Output:   {turn_output_tokens:5d} tokens ({turn_output_tokens/total_turn*100:.1f}%)")
                    if turn_thinking_tokens > 0:
                        print(f"   Thinking: {turn_thinking_tokens:5d} tokens ({turn_thinking_tokens/total_turn*100:.1f}%)")
                    
                    # ===== MODALITY DETAILS =====
                    print(f"\n🔬 MODALITY BREAKDOWN:")
                    print(f"{'─'*70}")
                    if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                        print(f"   Input Modalities:")
                        for detail in usage.prompt_tokens_details:
                            modality = detail.modality.name if hasattr(detail.modality, 'name') else str(detail.modality)
                            pct = (detail.token_count / turn_input_tokens * 100) if turn_input_tokens > 0 else 0
                            print(f"      • {modality:6s}: {detail.token_count:4d} tokens ({pct:5.1f}%)")
                    
                    if hasattr(usage, 'response_tokens_details') and usage.response_tokens_details:
                        print(f"   Output Modalities:")
                        for detail in usage.response_tokens_details:
                            modality = detail.modality.name if hasattr(detail.modality, 'name') else str(detail.modality)
                            pct = (detail.token_count / turn_output_tokens * 100) if turn_output_tokens > 0 else 0
                            print(f"      • {modality:6s}: {detail.token_count:4d} tokens ({pct:5.1f}%)")
                    
                    # ===== ADDITIONAL DETAILS =====
                    additional_info = []
                    if hasattr(usage, 'cached_content_token_count') and usage.cached_content_token_count:
                        additional_info.append(f"Cached: {usage.cached_content_token_count}")
                    if hasattr(usage, 'tool_use_prompt_token_count') and usage.tool_use_prompt_token_count:
                        additional_info.append(f"Tool-use: {usage.tool_use_prompt_token_count}")
                    
                    if additional_info:
                        print(f"\n   ℹ️  Other: {', '.join(additional_info)}")
                    
                    # ===== CUMULATIVE TOTALS =====
                    print(f"\n{'='*70}")
                    print(f"📈 CUMULATIVE SESSION TOTALS (All {self.turn_count} turns)")
                    print(f"{'='*70}")
                    grand_total = self.total_input_tokens + self.total_output_tokens + self.total_thinking_tokens
                    print(f"   Total Input:     {self.total_input_tokens:6d} tokens ({self.total_input_tokens/grand_total*100:.1f}%)")
                    print(f"   Total Output:    {self.total_output_tokens:6d} tokens ({self.total_output_tokens/grand_total*100:.1f}%)")
                    if self.total_thinking_tokens > 0:
                        print(f"   Total Thinking:  {self.total_thinking_tokens:6d} tokens ({self.total_thinking_tokens/grand_total*100:.1f}%)")
                    print(f"   ─" * 35)
                    print(f"   Grand Total:     {grand_total:6d} tokens")
                    print(f"{'='*70}\n")
                    
                    # Store for next turn delta calculation
                    self.previous_turn_input_tokens = turn_input_tokens

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
    
    def display_final_summary(self):
        """Display final token usage summary"""
        print("\n\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Total turns:        {self.turn_count}")
        print(f"Total input tokens: {self.total_input_tokens}")
        print(f"Total output tokens: {self.total_output_tokens}")
        if self.total_thinking_tokens > 0:
            print(f"Total thinking tokens: {self.total_thinking_tokens}")
        print(f"Grand total tokens: {self.total_input_tokens + self.total_output_tokens + self.total_thinking_tokens}")
        print("="*60 + "\n")

    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()

                send_text_task = tg.create_task(self.send_text())
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
        finally:
            # Display final token usage summary
            self.display_final_summary()


if __name__ == "__main__":
    main = AudioLoop()
    asyncio.run(main.run())