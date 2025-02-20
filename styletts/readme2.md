cd /workspace

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh


bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate

conda create -n tts  python=3.11.0
conda activate tts



git clone https://github.com/felipetrujillot/dataset


pip install click pandas matplotlib tensorboard SoundFile torchaudio munch torch pydub pyyaml librosa git+https://github.com/resemble-ai/monotonic_align.git


cd styletts

python train_first.py --config_path ./Configs/config.yml

python train_second.py --config_path ./Configs/config.yml

