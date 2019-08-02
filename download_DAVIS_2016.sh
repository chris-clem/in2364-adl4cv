#!/bin/sh

INITIAL_DIR=$(pwd)

mkdir DAVIS_2016
cd DAVIS_2016
wget https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
unzip DAVIS-data.zip
rm DAVIS-data.zip

cd $INITIAL_DIR