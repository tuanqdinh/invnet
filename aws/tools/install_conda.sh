#!/usr/bin/env bash
# setup Anaconda env
wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
bash Anaconda3-5.3.1-Linux-x86_64.sh -b -p ~/anaconda
rm Anaconda3-5.3.1-Linux-x86_64.sh
echo 'export PATH="~/anaconda3/bin:$PATH"' >> ~/.bashrc 

# Refresh basically
source .bashrc