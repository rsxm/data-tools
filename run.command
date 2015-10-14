#!/bin/bash

export PATH=~/anaconda/bin:$PATH
source activate datatools

DIR=$( cd "$( dirname "$0" )" && pwd )

cd "${DIR}"
echo ${PWD}

python extractor.py

read -s -n 1 -p "Press any key to exit..."
