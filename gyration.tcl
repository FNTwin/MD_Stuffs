set path [pwd]
puts -nonewline "\n Select the polymer (e.g fragment 1): "
gets stdin selmode
set sel [atomselect top "$selmode"]
#set sel [atomselect top "name Oc C2 C3"]
puts -nonewline "\n Name output file? : "
gets stdin fname
set outf [open "$path/$fname" w]
puts -nonewline "\n $path/$fname : "
set n [molinfo top get numframes]

for {set i 0} {$i < $n} {incr i} {
	molinfo top set frame $i
	set radgyr [measure rgyr $sel]
	puts $outf "$radgyr"
}
close $outf
puts "Done."
