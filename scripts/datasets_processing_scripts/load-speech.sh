#!/usr/bin/env bash
set -e

# incomplete script (not fully tested)
# 1. clone FedScale repo
# 2. download speech dataset
# 3. use python script to preprocess data
# /google_speech/npz/
#   - test/
#   - train/
#   - val/



# if [ ! -d FedScale ]; then
#   git clone https://github.com/SymbioticLab/FedScale.git
# fi

# chmod +x FedScale/benchmark/dataset/download.sh
# ./FedScale/benchmark/dataset/download.sh download speech

if ! command -v axel &>/dev/null; then
  echo "Installing axel for faster download"
  sudo apt-get install axel
fi

if [ ! -d google_speech ]; then
  echo "Downloading speech dataset"
  axel -a -n 10 https://fedscale.eecs.umich.edu/dataset/google_speech.tar.gz
  tar -xf google_speech.tar.gz
  rm -f google_speech.tar.gz
  mkdir -p google_speech/npz/
fi

if [ ! -d venv2 ]; then
  python3 -m pip install virtualenv
  python3 -m virtualenv venv2
  sudo apt-get update
  sudo apt-get -y install libsndfile1
  source venv2/bin/activate
  python3 -m pip install librosa==0.8 tensorflow==2.10.0 numpy~=1.21.6  
else
  source venv2/bin/activate
fi

script_path="$(dirname "${BASH_SOURCE[0]}")"
echo path="$script_path"
python3 "$script_path"/fedscale/process-google-speech-fedscale.py
