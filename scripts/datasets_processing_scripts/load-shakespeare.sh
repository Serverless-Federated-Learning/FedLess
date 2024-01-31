#!/usr/bin/env bash
set -e

if ! command -v wget &>/dev/null; then
  echo "wget not installed, aborting..."
  exit
fi

if ! command -v python3 &>/dev/null; then
  echo "python 3 not installed, aborting..."
  exit
fi

if ! command -v unzip &>/dev/null; then
  echo "unzip not installed, trying to install"
  if [[ $(sudo apt install -y unzip) -ne 0 ]]; then
    echo "Unzip could not be installed"
    exit
  fi
fi

if [ ! -d leaf ]; then
  git clone https://github.com/andreas-grafberger/leaf.git
fi
if [ ! -d venv ]; then
  python3 -m pip install virtualenv
  python3 -m virtualenv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate
cd leaf
python3 -m pip install --upgrade pip
python3 -m pip install -r <(sed '/^[tensorflow]/d' requirements.txt)
cd data/shakespeare

./preprocess.sh -s niid --sf 1.0 -k 64 -tf 0.9 -t sample --nochecksum
