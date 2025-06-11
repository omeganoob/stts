import argparse
import yaml
import uuid
import os
from stts import (
    load_model, preprocess_audio, transcribe_audio, translate_text,
    text_to_speech_google, text_to_speech_kokoro
)

def main():
    parser = argparse.ArgumentParser(description="SpeechToSpeech Testing hehe")
    parser.add_argument("audio_file", type=str)
    parser.add_argument("language", type=str, choices=["ja", "vi", "en"])
    parser.add_argument("dest_language", type=str, choices=["ja", "vi", "en"])
    parser.add_argument("tts", type=str, choices=["google", "kokoro"])
    parser.add_argument("--use_batch", type=str, choices=["yes", "no"], default="no")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    
    with open("env.yml", "r") as file:
        config = yaml.safe_load(file)

    model_japanese = load_model('zh-plus/faster-whisper-large-v2-japanese-5k-steps', args.device, (args.use_batch == "yes"))
    model_vietnamese = load_model('large-v3', args.device, (args.use_batch == "yes"))

    model = model_japanese if args.language == 'ja' else model_vietnamese

    preprocessed = preprocess_audio(args.audio_file)
    
    transcription = transcribe_audio(preprocessed, args.language, model, True, 16)

    target_language = args.dest_language
    
    translated_text = translate_text(transcription, target_language)
    print(f"Translated: {translated_text}")
    
    _file_name,_ = os.path.splitext(os.path.basename(args.audio_file))
    
    if not os.path.exists(f"translated/{target_language}/"):
        os.makedirs(f"translated/{target_language}/")
        
    output_file = f"translated/{target_language}/{_file_name}_{uuid.uuid4()}.wav"
    
    if args.tts == "google":
        text_to_speech_google(translated_text, target_language, output_file)
    elif args.tts == "kokoro":
        text_to_speech_kokoro(translate_text, target_language, output_file)

if __name__ == "__main__":
    main()