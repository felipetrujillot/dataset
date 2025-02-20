conda create -n tts  python=3.11.0


conda activate tts


pip install click pandas matplotlib tensorboard SoundFile torchaudio munch torch pydub pyyaml librosa git+https://github.com/resemble-ai/monotonic_align.git


python train_first.py --config_path ./Configs/config.yml


python train_second.py --config_path ./Configs/config.yml

