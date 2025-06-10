import argparse
from faster_whisper import WhisperModel
from googletrans import Translator
from gtts import gTTS
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import uuid
import requests
from faster_whisper import WhisperModel, BatchedInferencePipeline
from scipy.io import wavfile
import noisereduce
import os

def load_model(model_name, device, use_batched=False):
    compute_type = 'float16' if device == 'cuda' else 'float32'
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    if use_batched:
        model = BatchedInferencePipeline(model=model)
    return model

def preprocess_audio(file_path):
    _file_name,_ = os.path.splitext(os.path.basename(file_path))
    # load data
    rate, data = wavfile.read(file_path)
    # perform noise reduction
    reduced_noise = noisereduce.reduce_noise(y=data, sr=rate)
    
    if not os.path.exists(f"noise_reduce"):
        os.makedirs(f"noise_reduce")
    
    output_file = f"noise_reduce/{_file_name}_{uuid.uuid4()}.wav"
    wavfile.write(output_file, rate, reduced_noise)
    return output_file

def transcribe_audio(file_path, language, model, use_batched=False, batch_size=16):
    if use_batched:
        segments, info = model.transcribe(file_path, beam_size=5, language=language, batch_size=batch_size)
    else:
        segments, info = model.transcribe(file_path, beam_size=5, language=language)
        
    transcription = []
    for segment in segments:
        transcription.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    return " ".join([seg['text'] for seg in transcription])

def translate_text(text, target_language):
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text

def text_to_speech_google(text, lang, output_file):
    tts = gTTS(text=text, lang=lang)
    tts.save(output_file)
    print(f"Text-to-Speech output saved to {output_file}")

def text_to_speech_elevenlabs(text, voice_id, api_key, model, output_file):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        },
        "model_id": model, 
        "language_id": "vi"
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        with open(output_file, 'wb') as f:
            f.write(response.content)
        print(f"Text-to-Speech output saved to {output_file}")
    else:
        print(f"Error: {response.status_code}, {response.text}")
    # elevenlabs = ElevenLabs(
    #     api_key=api_key,
    # )
    
    # audio = elevenlabs.text_to_speech.convert(
    #     text=text,
    #     voice_id=voice_id,
    #     model_id=model,
    #     output_format="mp3_44100_128",
        
    #     voice_settings=VoiceSettings(
    #         stability=0.0,
    #         similarity_boost=1.0,
    #         style=0.0,
    #         use_speaker_boost=True,
    #         speed=1.0,
    #     ),
    # )
    
    # with open(output_file, 'wb') as f:
    #     for chunk in audio:
    #         if chunk:
    #             f.write(chunk)
    # print(f"{output_file}: A new audio file was saved successfully!")