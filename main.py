import argparse
from faster_whisper import WhisperModel
from googletrans import Translator
from gtts import gTTS
import yaml
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import uuid
import requests
from faster_whisper import WhisperModel, BatchedInferencePipeline
from scipy.io import wavfile
import noisereduce
import os

def load_model(model_name, device):
    compute_type = 'float16' if device == 'cuda' else 'float32'
    return WhisperModel(model_name, device=device, compute_type=compute_type)

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

def transcribe_audio(file_path, language, model):
    segments, info = model.transcribe(file_path, beam_size=5)
    
    print(f"Detected language: {info.language} with probability {info.language_probability}")
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
def main():
    parser = argparse.ArgumentParser(description="SpeechToSpeech Testing hehe")
    parser.add_argument("audio_file", type=str)
    parser.add_argument("language", type=str, choices=["ja", "vi"])
    parser.add_argument("tts", type=str, choices=["google", "elevenlabs"])
    parser.add_argument("--use_batch", type=str, choices=["yes", "no"], default="no")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    
    with open("env.yml", "r") as file:
        config = yaml.safe_load(file)

    model_japanese = load_model('zh-plus/faster-whisper-large-v2-japanese-5k-steps', args.device)
    model_vietnamese = load_model('distil-large-v3', args.device)

    model = model_japanese if args.language == 'ja' else model_vietnamese
    if args.use_batch == "yes":
        print("\tUse batch model.\n");
        model = BatchedInferencePipeline(model=model)

    preprocessed = preprocess_audio(args.audio_file)
    
    transcription = transcribe_audio(preprocessed, args.language, model)

    target_language = "vi" if args.language == "ja" else "ja"
    
    translated_text = translate_text(transcription, target_language)
    print(f"Translated: {translated_text}")
    
    _file_name,_ = os.path.splitext(os.path.basename(args.audio_file))
    
    if not os.path.exists(f"translated/{target_language}/"):
        os.makedirs(f"translated/{target_language}/")
        
    output_file = f"translated/{target_language}/{_file_name}_{uuid.uuid4()}.wav"
    
    if args.tts == "google":
        text_to_speech_google(translated_text, target_language, output_file)
    elif args.tts == "elevenlabs":
        if not config['elevenlabs']['api_key'] or not config['elevenlabs']['voice_id']:
            raise ValueError("ElevenLabs API key and Voice ID are required for ElevenLabs TTS")
        text_to_speech_elevenlabs(
            translated_text, 
            config['elevenlabs']['voice_id'],
            config['elevenlabs']['api_key'],
            config['elevenlabs']['model'],
            output_file
        )

if __name__ == "__main__":
    main()
    """
    meian/meian_0000.wav
    >|この前探った時は、途中に瘢痕の隆起があったので、ついそこが行きどまりだとばかり思って、ああ云ったんですが、
    >|kono mae sagut ta toki wa 、 tochu- ni hankon no ryu-ki ga at ta node 、 tsui soko ga 
    yukidomari da to bakari omot te 、 a- yut ta n desu ga
    """
    
    """
    elevenlabs:
        api_key: "sk_07b89be07a583ad97ce60ddb1dac604f881d4d81ac847688"
        model: "eleven_flash_v2_5"
        voice_id: "JBFqnCBsd6RMkjVDRZzb"
    """