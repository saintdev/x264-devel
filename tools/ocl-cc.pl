#!/usr/bin/env perl
# this strips excess whitespace and # <file> lines left from cpp
# and puts program source in a string

$ARGV[0] =~ /([\w\d_]+)\.(cl|il)/;
$processed_src .= qq(const char *x264_opencl_$1_src = ");

open(CLFILE, "<", $ARGV[0]) || die "Error opening $ARGV[0] for reading";

$DEBUG = "YES";

while(<CLFILE>) {
    s/\"/\\"/g;       # escape quotations
    if ($DEBUG eq "YES") {
        s/[\n\r]/\\n"\n"/g;
    } else {
        s/[\n\r]+/\\n"\n"/g;    # condense newlines into a single newline.
        s/\s+/ /g;      # condense all space into ' '
        s/^\s+//g;      # remove all spaces at the beginning of lines
        # remove spaces that don't separate identifiers (alphanumeric plus _)
        s/([^\w\d_]) (\S)/$1$2/g;
        s/(\S) ([^\w\d_])/$1$2/g;
    }
    $processed_src .= $_;
}
$processed_src .= qq(";\n);

close(CLFILE) or exit 1;
open(CLFILE, ">", $ARGV[0]) or die "Error opening $ARGV[0] for writing";
print CLFILE $processed_src;
close(CLFILE) or exit 1;
