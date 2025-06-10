import argparse
import os
import csv
from Levenshtein import distance as levenshtein_distance
from stts import (
    load_model, preprocess_audio, transcribe_audio, translate_text,
    text_to_speech_google, text_to_speech_elevenlabs
)

def evaluate_transcription(transcription, ground_truth):
    # Calculate the Levenshtein distance between the transcribed text and the ground truth
    distance = levenshtein_distance(transcription, ground_truth)
    
    # Calculate the similarity score (1 - normalized distance)
    max_length = max(len(transcription), len(ground_truth))
    similarity = 1 - (distance / max_length) if max_length > 0 else 0
    
    return similarity

def main_eva():
    parser = argparse.ArgumentParser(description="Transcribe audio files and evaluate accuracy")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing audio file information")
    parser.add_argument("audio_dir", type=str, help="Directory containing the audio files")
    parser.add_argument("device", type=str, choices=["cpu", "cuda"], help="Device to run the model on")
    args = parser.parse_args()
    
    # model = load_model('zh-plus/faster-whisper-large-v2-japanese-5k-steps', args.device)
    model = load_model('large-v3', args.device)
    
    with open(args.csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='|')
        for row in reader:
            file_name, japanese_text, romaji_text, score = row
            file_path = os.path.join(args.audio_dir, file_name)
            
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist. Skipping...")
                continue
            
            # preprocessed_file = preprocess_audio(file_path)
            transcription = transcribe_audio(file_path, 'ja', model)
            accuracy = evaluate_transcription(transcription, japanese_text)
            
            print(f"File: {file_name}")
            print(f"Transcription: {transcription}")
            print(f"Ground Truth: {japanese_text}")
            print(f"Accuracy: {accuracy:.2f}\n")

if __name__ == "__main__":
    main_eva()