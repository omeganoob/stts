from faster_whisper import WhisperModel
from googletrans import Translator
from gtts import gTTS
import uuid
from faster_whisper import WhisperModel, BatchedInferencePipeline
from scipy.io import wavfile
import noisereduce
import os
from kokoro import KPipeline
import soundfile as sf
import numpy as np

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
        segments, info = model.transcribe(file_path, beam_size=5, language=language, condition_on_previous_text=False, batch_size=batch_size)
    else:
        segments, info = model.transcribe(file_path, beam_size=5, language=language, condition_on_previous_text=False)
        
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

def text_to_speech_kokoro(text, language, output_file):
    """
    Convert Japanese text to speech using Kokoro TTS model
    
    Args:
        text (str): Japanese text to convert to speech
        language (str): Language code ('ja' for Japanese)
        output_file (str): Path to save the output WAV file
    """
    try:
        if language == 'ja':
            language = 'j'
        # Initialize Kokoro model
        kokoro = KPipeline(lang_code=language)
        generator = kokoro(text, voice='af_heart')
        # Generate speech from text
        for i, (gs, ps, audio) in enumerate(generator):
            print(i, gs, ps)
            sf.write(output_file, audio, 24000)
    except Exception as e:
        print(f"Error generating speech with Kokoro TTS: {e}")