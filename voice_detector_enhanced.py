import pyaudio
import wave
import numpy as np
import time
from datetime import datetime
import os
import speech_recognition as sr
from pydub import AudioSegment
from pydub.effects import normalize
import noisereduce as nr


class EnhancedVoiceDetector:
    """
    Enhanced Voice Activity Detector with:
    - Background noise reduction
    - Real speech-to-text transcription
    - Auto-stop after 10 seconds of silence
    """
    
    def __init__(
        self,
        silence_threshold=800,
        silence_duration=10.0,
        sample_rate=16000,
        chunk_size=1024,
        channels=1,
        output_dir="voice_recordings",
        reduce_noise=True,
        auto_transcribe=True
    ):
        """
        Initialize Enhanced Voice Detector.
        
        Args:
            silence_threshold: RMS threshold for silence detection
            silence_duration: Seconds of silence before auto-stop (default 10s)
            sample_rate: Audio sample rate
            chunk_size: Audio chunk size
            channels: Number of audio channels
            output_dir: Directory to save recordings
            reduce_noise: Enable noise reduction
            auto_transcribe: Automatically transcribe recordings
        """
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.output_dir = output_dir
        self.reduce_noise = reduce_noise
        self.auto_transcribe = auto_transcribe
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.is_recording = False
        self.audio_frames = []
        self.transcriptions = []
        self.last_speech_time = None
        self.speech_detected = False
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Speech recognizer
        if auto_transcribe:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
        
    def calculate_rms(self, audio_chunk):
        """Calculate RMS of audio chunk."""
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        
        if len(audio_data) == 0:
            return 0.0
        
        audio_float = audio_data.astype(np.float32)
        rms = np.sqrt(np.mean(audio_float**2))
        return rms
    
    def is_speech(self, audio_chunk):
        """Determine if audio contains speech."""
        rms = self.calculate_rms(audio_chunk)
        return rms > self.silence_threshold
    
    def apply_noise_reduction(self, audio_file):
        """
        Apply noise reduction to recorded audio.
        
        Args:
            audio_file: Path to WAV file
            
        Returns:
            Path to cleaned audio file
        """
        print("🔇 Applying noise reduction...")
        
        try:
            # Load audio
            audio_data, sample_rate = self._load_wav(audio_file)
            
            # Apply noise reduction
            # Use first 0.5 seconds as noise profile
            noise_sample = audio_data[:int(0.5 * sample_rate)]
            reduced_noise = nr.reduce_noise(
                y=audio_data,
                sr=sample_rate,
                y_noise=noise_sample,
                prop_decrease=0.8,  # Reduce noise by 80%
                stationary=False
            )
            
            # Normalize audio
            reduced_noise = reduced_noise / np.max(np.abs(reduced_noise)) * 0.9
            
            # Save cleaned audio
            cleaned_file = audio_file.replace('.wav', '_cleaned.wav')
            self._save_wav(cleaned_file, reduced_noise, sample_rate)
            
            print(f"✅ Noise reduced: {cleaned_file}")
            return cleaned_file
            
        except Exception as e:
            print(f"⚠️  Noise reduction failed: {e}")
            return audio_file
    
    def _load_wav(self, filename):
        """Load WAV file as numpy array."""
        with wave.open(filename, 'rb') as wf:
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            # Normalize to -1 to 1 range
            audio_data = audio_data / 32768.0
            return audio_data, sample_rate
    
    def _save_wav(self, filename, audio_data, sample_rate):
        """Save numpy array as WAV file."""
        # Convert back to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
    
    def transcribe_audio(self, audio_file):
        """
        Transcribe audio file to text using Google Speech Recognition.
        
        Args:
            audio_file: Path to WAV file
            
        Returns:
            Transcribed text or None if failed
        """
        print("🎯 Transcribing audio...")
        
        try:
            with sr.AudioFile(audio_file) as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Record the audio
                audio_data = self.recognizer.record(source)
                
                # Recognize speech using Google Speech Recognition
                text = self.recognizer.recognize_google(audio_data)
                
                print(f"✅ Transcription: {text}")
                return text
                
        except sr.UnknownValueError:
            print("⚠️  Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"⚠️  Transcription service error: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Transcription failed: {e}")
            return None
    
    def start_recording(self):
        """Start recording with voice activity detection."""
        print("🎤 Enhanced Voice Detector started")
        print(f"📊 Silence Threshold: {self.silence_threshold}")
        print(f"⏱️  Auto-stop: {self.silence_duration}s of silence")
        print(f"🔇 Noise Reduction: {'Enabled' if self.reduce_noise else 'Disabled'}")
        print(f"📝 Auto Transcribe: {'Enabled' if self.auto_transcribe else 'Disabled'}")
        print("\n🔊 Waiting for speech...\n")
        
        self.is_recording = True
        self.audio_frames = []
        self.last_speech_time = None
        self.speech_detected = False
        
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        try:
            while self.is_recording:
                audio_chunk = self.stream.read(self.chunk_size, exception_on_overflow=False)
                
                if self.is_speech(audio_chunk):
                    current_time = time.time()
                    
                    if not self.speech_detected:
                        self.speech_detected = True
                        print("✅ Speech detected! Recording started...")
                    
                    self.last_speech_time = current_time
                    self.audio_frames.append(audio_chunk)
                    
                else:
                    # Silence detected
                    if self.speech_detected:
                        self.audio_frames.append(audio_chunk)
                        
                        if self.last_speech_time is not None:
                            silence_elapsed = time.time() - self.last_speech_time
                            
                            if silence_elapsed >= self.silence_duration:
                                print(f"\n⏹️  No speech for {self.silence_duration}s - Stopping...")
                                self.stop_recording()
                                break
                            
                            # Show countdown
                            if silence_elapsed >= (self.silence_duration - 5):
                                remaining = self.silence_duration - silence_elapsed
                                print(f"⏱️  Silence: {remaining:.1f}s remaining...", end='\r')
                
        except KeyboardInterrupt:
            print("\n⚠️  Recording interrupted")
            self.stop_recording()
    
    def stop_recording(self):
        """Stop recording and process audio."""
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio_frames and self.speech_detected:
            # Save raw recording
            raw_file = self.save_recording()
            print(f"💾 Raw recording: {raw_file}")
            
            # Apply noise reduction if enabled
            if self.reduce_noise:
                cleaned_file = self.apply_noise_reduction(raw_file)
            else:
                cleaned_file = raw_file
            
            # Transcribe if enabled
            if self.auto_transcribe:
                transcription = self.transcribe_audio(cleaned_file)
                if transcription:
                    self.add_transcription(transcription, "User")
                    self.save_transcriptions()
            
            return cleaned_file
        else:
            print("\n⚠️  No speech detected")
            return None
    
    def save_recording(self):
        """Save recorded audio to WAV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"recording_{timestamp}.wav")
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(self.audio_frames))
        
        return filename
    
    def add_transcription(self, text, speaker="User"):
        """Store transcription."""
        transcription = {
            "timestamp": datetime.now().isoformat(),
            "speaker": speaker,
            "text": text
        }
        self.transcriptions.append(transcription)
        print(f"\n📝 Transcribed: [{speaker}] {text}")
    
    def get_transcriptions(self):
        """Get all transcriptions."""
        return self.transcriptions
    
    def save_transcriptions(self, filename=None):
        """Save transcriptions to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"transcription_{timestamp}.txt")
        
        with open(filename, 'w') as f:
            for trans in self.transcriptions:
                f.write(f"[{trans['timestamp']}] {trans['speaker']}: {trans['text']}\n")
        
        print(f"💾 Transcription saved: {filename}")
        return filename
    
    def cleanup(self):
        """Clean up resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("\n🧹 Cleaned up")


def main():
    """Main function."""
    print("\n🎤 Enhanced Voice Activity Detector")
    print("=" * 60)
    print("Features:")
    print("  ✅ Background noise reduction")
    print("  ✅ Real-time speech-to-text transcription")
    print("  ✅ Auto-stop after 10 seconds of silence")
    print("=" * 60)
    
    # Create detector
    detector = EnhancedVoiceDetector(
        silence_threshold=800,      # Adjust if needed
        silence_duration=5.0,      # 10 seconds
        reduce_noise=True,          # Enable noise reduction
        auto_transcribe=True        # Enable auto transcription
    )
    
    try:
        # Start recording
        detector.start_recording()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        detector.cleanup()


if __name__ == "__main__":
    main()
