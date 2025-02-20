# Converter


Herramienta para convertir .mp3 a chunks


1. Preprocess the Audio
Convert the audio to a common format (e.g., WAV, FLAC, MP3).
Ensure a consistent sample rate (e.g., 16kHz for speech models).
Use ffmpeg to convert:

ffmpeg -i audio.mp3 -ac 1 -ar 24000 output.wav


2. main.py


3. converter


