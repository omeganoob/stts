import pyaudio
import wave
import argparse
import os
import uuid
from stts import load_model, transcribe_audio, preprocess_audio

# Function to record audio from the microphone
def record_audio(pyaudio_p, stream, filename, duration=5, chunk=1024, sample_format=pyaudio.paInt16, channels=1, fs=16000):
    frames = []
    for i in range(0, int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(pyaudio_p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()
    
def pcm2wav(pcm_data, filename, sample_rate=16000, sample_width=2, channels=1):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
        
def main():
    parser = argparse.ArgumentParser(description="Real-time audio transcription")
    parser.add_argument("--language", type=str, choices=["ja", "vi", "en"], default="ja")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    model = load_model('large-v3', args.device)
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open the microphone stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)

    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            # Record a short audio segment
            temp_filename = f"temp/temp_audio.wav"
            record_audio(p, stream, temp_filename, duration=5)
            # Transcribe the recorded audio
            transcription = transcribe_audio(temp_filename, args.language, model)
            print(f"Transcription: {transcription}")
            # Remove the temporary audio file
            os.remove(temp_filename)
    except KeyboardInterrupt:
        print("Stopped listening.")
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()