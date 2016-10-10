#!/bin/bash

# Runs runit.sh on every input file in the input directory
for file in input/*; do
	echo $file
	./runit.sh $file
done
