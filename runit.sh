#!/bin/bash

# Attempts to induce the senses of a given target word given many contexts with the target word. The contexts should have at least 25 words to both the left and right of the target word for the system to work well. 

# Assertions about the input file:
# - The file is in senseval2 xml format
# - The file name is of the form: <target-word>-<part-of-speech>-*.xml

# TODO: what the script will do

if [ -z $1 ]
then
	echo "usage: ./run_it.sh <senseval2-xml-file>"
	echo "Where the input file's name is in the format:"
	echo "<target-word>-<part-of-speech>.xml"
	exit 1
fi

# remove any input files from the last time we ran
rm hdp-wsi/wsi_input/example/all/* &> /dev/null

# writeToHDP.py converts the given senseval2 xml file, and converts it into a format that can be processed by hdp-wsi. In addition, this script removes stop words from the contexts, and it converts everything to lowercase because the hdp-wsi tool does not do that for whatever reason. 
echo "Converting $1 to hdp-wsi format"
python writeToHDP.py $1

cd hdp-wsi
# Run word sense induction tool on the contexts we just gave it
echo "Running word sense induction"
./run_wsi.sh &> /dev/null
cd ..

# Read the results from the word sense induction tool, and generate definition
echo "Processing results from hdp-wsi"
python postProcessing.py $1 &> /dev/null

cd senseclusters_scorer/
# calculate the precision of our clustering
echo "Calculating sense cluster scores"
./senseclusters_scorer.sh key answers &> /dev/null
# print out the scores
cat report.out
cd ..
