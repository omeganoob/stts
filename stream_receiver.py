import pyaudio
import wave
import argparse
import os
import uuid
from stts import load_model, transcribe_audio
import numpy as np
import noisereduce 

#Audio sample data
RATE = 16000
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16

def pcm2wav(pcm_data, filename, pyaudio_p, channels=1):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio_p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(pcm_data)

def pcm2numpy(pcm_data):
    audio_array = np.frombuffer(pcm_data, np.int16).flatten().astype(np.float32) / 32768.0
    return audio_array

def record_audio(stream, duration=5):
    frames = []
    for i in range(0, int(RATE / CHUNK_SIZE * duration)):
        data = stream.read(CHUNK_SIZE)
        frames.append(data)
    
    return b''.join(frames)

def is_silence(audio_data, silence_threshold=0.01, silence_duration_ratio=0.8):
    """
    Detect if audio is mostly silence
    Args:
        audio_data: numpy array of audio samples
        silence_threshold: amplitude threshold below which audio is considered silence
        silence_duration_ratio: ratio of samples that must be below threshold to consider it silence
    """
    rms = np.sqrt(np.mean(audio_data**2))
    
    quiet_samples = np.sum(np.abs(audio_data) < silence_threshold)
    quiet_ratio = quiet_samples / len(audio_data)
    return rms < silence_threshold * 0.5 or quiet_ratio > silence_duration_ratio

def process_audio_chunk(model, audio_data, language):
    # no audio
    if audio_data.size == 0:
        return ""
    # silence
    if is_silence(audio_data):
        return ""
    # too quite
    if np.max(np.abs(audio_data)) < 5e-3:
        return ""
    
    reduced_noise_audio = noisereduce.reduce_noise(y=audio_data, sr=RATE)
    transcription = transcribe_audio(reduced_noise_audio, language, model)
    return transcription
        
def start_transcribe(model, lang):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=1,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)
    print("Listening... Press Ctrl+C to stop.")
    try:
        while True:
            pcm_buffer = record_audio(stream, duration=2)
            
            """Read to temporary wav file"""
            # temp_filename = f"temp/temp_audio.wav"
            # pcm2wav(pcm_buffer, temp_filename, p)
            # transcription = transcribe_audio(temp_filename, lang, model)
            # os.remove(temp_filename)

            """Read to numpy array"""
            audio_data = pcm2numpy(pcm_buffer)
            transcription = process_audio_chunk(model, audio_data, lang)
    except KeyboardInterrupt:
        print("Stopped listening.")
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

def main():
    parser = argparse.ArgumentParser(description="Real-time audio transcription")
    parser.add_argument("--language", type=str, choices=["ja", "vi", "en"], default="ja")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    # model = load_model('./models/faster-whisper-large-v3', args.device)
    model = load_model('large-v3', args.device)
    
    start_transcribe(model, args.language)

if __name__ == "__main__":
    main()