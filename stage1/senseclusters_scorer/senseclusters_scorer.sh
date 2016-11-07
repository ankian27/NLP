#!/bin/csh

if ($#argv != 2) then
   echo "USAGE ERROR: senseclusters_scorer.sh ANSWERS KEY"
   echo "  ANSWERS is a file of system generated answers in senseval format"
   echo "  KEY is a gold standard file in senseval format"
   exit 1
endif

set answers = $1
set key = $2

# this program will convert a senseval-4 answer file into cluto
# formatted output 

perl clutogen.pl $answers > $answers.cluto.out

# this program will convert a senseval-4 key file into a senseclusters
# key file

perl scgen.pl $key > $key.sc.out

# the following programs are all taken from the SenseClusters package
# http://senseclusters.sourceforge.net

# create a cluster (answers) by sense (key) matrix

perl cluto2label.pl $answers.cluto.out $key.sc.out > cluto2label.out

# find the optimal mapping of clusters to senses

perl label.pl cluto2label.out > label.out

# generate a report showing results

perl report.pl label.out cluto2label.out > report.out


