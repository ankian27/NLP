# this program converts a senseval key file into a senseclusters 
# formatted key file

while (<>) {
	@line = split;
	print "<instance id=\"$line[1]\"\/> <sense id=\"$line[2]\"\/>\n"; }

