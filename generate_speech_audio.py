"""
Generate a simple audio file with speech "Hello, how are you?"
Using gTTS and saving as MP3 (which Gemini API accepts)
"""
from gtts import gTTS

print("Generating audio file with 'Hello, how are you?'")

text = "Hello, how are you?, Who is virat kohli?"

# Create speech
tts = gTTS(text=text, lang='en', slow=False)

# Save as MP3
tts.save("hello_speech.mp3")

print("✓ Audio file created: hello_speech.mp3")
print("\nYou can now test with:")
print("python3 test1.py hello_speech.mp3")
