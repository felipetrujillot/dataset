import os
import whisper
import csv
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
_ESPEAK_LIBRARY = '/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.dylib'  #use the Path to the library.
EspeakWrapper.set_library(_ESPEAK_LIBRARY)

# Load Whisper model (choose: tiny, base, small, medium, large)
model = whisper.load_model("medium")

# Define input/output paths
chunks_folder = "chunks"
output_csv = "transcriptions.csv"

# Collect all audio files in the folder
audio_files = sorted([f for f in os.listdir(chunks_folder) if f.endswith(".wav")])

# Open CSV file for writing
with open(output_csv, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file, delimiter="|")  # Use '|' as delimiter
    #writer.writerow(["FILENAME", "TRANSCRIPT"])  # Write header

    # Transcribe each file
    for audio_file in audio_files:
        file_path = os.path.join(chunks_folder, audio_file)
        result = model.transcribe(file_path, language="es")
        
        # Convert transcript to IPA
        ipa_transcript = phonemize(result["text"], language="es", backend="espeak", strip=True)


        # Write to CSV
        writer.writerow([f"chunks/{audio_file}", ipa_transcript])

print(f"Transcriptions saved to {output_csv}")
