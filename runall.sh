#!/bin/bash

for file in input/*; do
	echo $file
	./runit.sh $file | grep Precision
done
