.#!/bin/bash

wget https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tgz
tar xvzf Python-3.6.0.tgz
cd Python-3.6.0
./configure --prefix=$HOME/Python-3.6.0 --enable-optimizations
make
make install

cd ..

echo 'export PATH=$HOME/Python-3.6.0/:$PATH' >> ~/.bash_profile
echo 'export PYTHONPATH=$HOME/Python-3.6.0/' >> ~/.bash_profile

source ~/.bash_profile

git clone https://github.com/siddsach/Interpreting-Attention

pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl

cd Interpreting-Attention
pip3 install -r requirements.txt
python -m spacy download en

python get_data.py
python experiments.py
