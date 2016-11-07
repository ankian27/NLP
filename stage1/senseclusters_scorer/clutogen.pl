# this program will read in a solution file in the form
#
# word word.instanceid word.cluster
#
# and output a file where each word.cluster is assigned
# a unique integer. 
# 
# the output from this program can then be used in the
# senseclusters program cluto2label.pl to create an 
# evaluation matrix

$clusterid = 0;
while (<>) {
	@line = split;
	if (not exists($answers{$line[2]})) {
		$answers{$line[2]} = $clusterid;
		print "$answers{$line[2]}\n";
		$clusterid++;
	}
	else {
		print "$answers{$line[2]}\n";
	}
}
