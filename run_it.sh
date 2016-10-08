#!/bin/bash

if [ -z $1 ]
then
	echo "usage: ./run_it.sh <senseval2-xml-file>"
	echo "Where the input file's name is in the format:"
	echo "<target-word>-<part-of-speech>.xml"
	exit 1
fi

rm hdp-wsi/wsi_input/example/all/*

python writeToHDP.py $1

cd hdp-wsi
./run_wsi.sh
cd ..

python postProcessing.py $1
