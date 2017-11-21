.#!/bin/bash


#Download python3 from source and install
#wget https://www.python.org/ftp/python/3.6.0/Python-3.6.0.tgz
#tar xvzf Python-3.6.0.tgz
#cd Python-3.6.0
#./configure --prefix=$HOME/Python-3.6.0 --enable-optimizations
#make
#make install

cd ..


#Install bz2 and remake python after doing so
cd py-360
cd bz...
make
make -f Makefile-libbz2_so
make install PREFIX=$HOME/py-360/Python-3.6.0

#Install python
cd ..
cd Python-3.6.0
./configure --prefix=$HOME/py-360/Python-3.6.0 --enable-optimizations
make install

#Update PATH and PYTHONPATH variables
echo 'export PATH=$HOME/py-360/Python-3.6.0/:$PATH' >> ~/.bash_profile
echo 'export PYTHONPATH=$HOME/py-360/Python-3.6.0/' >> ~/.bash_profile

source ~/.bash_profile

#get get repo
#git clone https://github.com/siddsach/Interpreting-Attention

#install torch
#pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl

cd Interpreting-Attention
#install dependencies
pip install -r requirements.txt
python -m spacy download en

#run job
python get_data.py
python experiments.py
