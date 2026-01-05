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
    def __init__(self, max_turns=3):
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
        
        # History management
        self.max_turns = max_turns
        self.conversation_history = []

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            elif text.lower() == "history":
                self.print_history()
                continue
            elif text.lower() == "stats":
                self.print_stats()
                continue
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
            user_text = ""
            model_text = ""
            
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                if (
                    server_content := response.server_content
                ) and server_content.input_transcription:
                    if self.last_speaker != "User":
                        print("\nUser: ", end="")
                        self.last_speaker = "User"
                    text_chunk = server_content.input_transcription.text
                    print(text_chunk, end="", flush=True)
                    user_text += text_chunk
                if (
                    server_content := response.server_content
                ) and server_content.output_transcription:
                    if self.last_speaker != "Model":
                        print("\nModel: ", end="")
                        self.last_speaker = "Model"
                    text_chunk = server_content.output_transcription.text
                    print(text_chunk, end="", flush=True)
                    model_text += text_chunk
                
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

            # Save to history
            if user_text:
                self.conversation_history.append({
                    "role": "user",
                    "text": user_text.strip(),
                    "turn": self.turn_count + 1,
                    "tokens": turn_input_tokens
                })
            if model_text:
                self.conversation_history.append({
                    "role": "model",
                    "text": model_text.strip(),
                    "turn": self.turn_count + 1,
                    "tokens": turn_output_tokens
                })

            # Update total counters after turn completes
            if turn_input_tokens > 0 or turn_output_tokens > 0:
                self.total_input_tokens += turn_input_tokens
                self.total_output_tokens += turn_output_tokens
                self.total_thinking_tokens += turn_thinking_tokens
                self.turn_count += 1
                
                # Get the last usage metadata for detailed display
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    usage = response.usage_metadata
                    
                    # Print comprehensive token usage for this turn
                    print(f"\n\n{'='*60}")
                    print(f"Turn #{self.turn_count} Token Usage:")
                    print(f"{'='*60}")
                    
                    # Main token counts
                    print(f"📥 Input tokens:     {turn_input_tokens}")
                    print(f"📤 Output tokens:    {turn_output_tokens}")
                    if turn_thinking_tokens > 0:
                        print(f"🧠 Thinking tokens:  {turn_thinking_tokens}")
                    
                    # Total for this turn
                    if hasattr(usage, 'total_token_count') and usage.total_token_count:
                        print(f"📊 Total this turn:  {usage.total_token_count}")
                    
                    # Additional token details
                    additional_info = []
                    if hasattr(usage, 'cached_content_token_count') and usage.cached_content_token_count:
                        additional_info.append(f"Cached: {usage.cached_content_token_count}")
                    if hasattr(usage, 'tool_use_prompt_token_count') and usage.tool_use_prompt_token_count:
                        additional_info.append(f"Tool-use: {usage.tool_use_prompt_token_count}")
                    
                    if additional_info:
                        print(f"ℹ️  Other: {', '.join(additional_info)}")
                    
                    # Modality breakdown for input
                    if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                        print(f"\n📝 Input Modality Breakdown:")
                        for detail in usage.prompt_tokens_details:
                            modality = detail.modality.name if hasattr(detail.modality, 'name') else str(detail.modality)
                            print(f"   - {modality}: {detail.token_count} tokens")
                    
                    # Modality breakdown for output
                    if hasattr(usage, 'response_tokens_details') and usage.response_tokens_details:
                        print(f"📝 Output Modality Breakdown:")
                        for detail in usage.response_tokens_details:
                            modality = detail.modality.name if hasattr(detail.modality, 'name') else str(detail.modality)
                            print(f"   - {modality}: {detail.token_count} tokens")
                    
                    # Cumulative totals
                    print(f"\n{'─'*60}")
                    print(f"📈 Cumulative Token Usage:")
                    print(f"{'─'*60}")
                    print(f"Total Input:     {self.total_input_tokens}")
                    print(f"Total Output:    {self.total_output_tokens}")
                    if self.total_thinking_tokens > 0:
                        print(f"Total Thinking:  {self.total_thinking_tokens}")
                    print(f"Grand Total:     {self.total_input_tokens + self.total_output_tokens + self.total_thinking_tokens}")
                    
                    # Warning about history size
                    if self.turn_count >= self.max_turns:
                        print(f"\n⚠️  WARNING: Reached {self.max_turns} turns!")
                        print(f"   History is growing large. Consider restarting (type 'q').")
                    elif self.turn_count >= self.max_turns * 0.75:
                        print(f"\nℹ️  FYI: {self.turn_count}/{self.max_turns} turns used")
                    
                    print(f"{'='*60}\n")

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
    
    def print_history(self):
        """Print conversation history"""
        print("\n\n" + "="*60)
        print("📜 CONVERSATION HISTORY")
        print("="*60)
        print(f"Total messages: {len(self.conversation_history)}")
        print(f"Total turns: {self.turn_count}")
        print("="*60)
        for msg in self.conversation_history:
            role_icon = "👤" if msg['role'] == 'user' else "🤖"
            text_preview = msg['text'][:80] + "..." if len(msg['text']) > 80 else msg['text']
            print(f"\n{role_icon} Turn {msg['turn']} | {msg['role'].upper()}")
            print(f"   {text_preview}")
            if 'tokens' in msg and msg['tokens']:
                print(f"   Tokens: {msg['tokens']}")
        print("\n" + "="*60 + "\n")
    
    def print_stats(self):
        """Print session statistics"""
        print("\n\n" + "="*60)
        print("📊 SESSION STATISTICS")
        print("="*60)
        print(f"Turns completed:     {self.turn_count}/{self.max_turns}")
        print(f"Messages in history: {len(self.conversation_history)}")
        print(f"Total input tokens:  {self.total_input_tokens}")
        print(f"Total output tokens: {self.total_output_tokens}")
        print(f"Total thinking:      {self.total_thinking_tokens}")
        print(f"Grand total:         {self.total_input_tokens + self.total_output_tokens + self.total_thinking_tokens}")
        
        if self.turn_count > 0:
            avg_input = self.total_input_tokens / self.turn_count
            avg_output = self.total_output_tokens / self.turn_count
            print(f"\nAverage per turn:")
            print(f"  Input:  {avg_input:.0f} tokens")
            print(f"  Output: {avg_output:.0f} tokens")
        
        print("="*60 + "\n")
    
    def display_final_summary(self):
        """Display final token usage summary"""
        print("\n\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Total turns:        {self.turn_count}")
        print(f"Total messages:     {len(self.conversation_history)}")
        print(f"Total input tokens: {self.total_input_tokens}")
        print(f"Total output tokens: {self.total_output_tokens}")
        if self.total_thinking_tokens > 0:
            print(f"Total thinking tokens: {self.total_thinking_tokens}")
        print(f"Grand total tokens: {self.total_input_tokens + self.total_output_tokens + self.total_thinking_tokens}")
        print("="*60 + "\n")

    async def run(self):
        try:
            print(f"\n{'='*60}")
            print(f"🎙️  Voice Agent with History Management")
            print(f"{'='*60}")
            print(f"Commands:")
            print(f"  'q' - Quit")
            print(f"  'history' - View conversation history")
            print(f"  'stats' - View session statistics")
            print(f"\nMax turns before warning: {self.max_turns}")
            print(f"{'='*60}\n")
            
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
    # Set max_turns limit (default: 20)
    main = AudioLoop(max_turns=3)
    asyncio.run(main.run())
