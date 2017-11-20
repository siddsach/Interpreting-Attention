.#!/bin/bash

wget https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tgz
tar xvzf Python-3.6.0.tgz
cd Python-3.6.0
./configure --prefix=$HOME/py-360 --enable-optimizations
make
make install

cd ..

git clone https://github.com/siddsach/Interpreting-Attention

mv Interpreting-Attention/.bash_profile .
source ~/.bash_profile

pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl

cd Interpreting-Attention
pip3 install -r requirements.txt
python3 -m spacy download en

python3 get_data.py
python3 experiments.py
