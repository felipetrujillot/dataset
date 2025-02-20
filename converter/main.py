from pydub import AudioSegment
from pydub.silence import split_on_silence

audio = AudioSegment.from_wav("output.wav")
chunks = split_on_silence(audio, min_silence_len=700, silence_thresh=-40)

for i, chunk in enumerate(chunks):
    chunk.export(f"chunk_{i}.wav", format="wav")
