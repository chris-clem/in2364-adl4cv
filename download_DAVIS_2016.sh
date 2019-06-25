#!/bin/sh

# This scripts downloads the DAVIS data and unzips it.
# Adapted from https://github.com/davisvideochallenge/davis/blob/master/data/get_davis.sh

FILE=DAVIS-data.zip
URL=https://graphics.ethz.ch/Downloads/Data/Davis
CHECKSUM=0cb3cf9c5617209fa3cc4794e52a2ffa

INITIAL_DIR=$(pwd)

mkdir DAVIS_2016
cd DAVIS_2016
wget https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
unzip DAVIS-data.zip
rm DAVIS-data.zip

cd $INITIAL_DIR