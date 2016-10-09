# this program creates random results!! The number of possible clusters 
# is set as an upper bound on the random number generator

$upperlimit = 4; 

# generate upperlimit random clusters from 0 to upperlimit-1

while (<>) {
	@line = split;
	$random = int(rand($upperlimit));
	print "$line[0] $line[1] $line[0].$random\n";
}

