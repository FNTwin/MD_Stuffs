set path [pwd]
puts -nonewline "\n Select the polymer (e.g fragment 1): "
gets stdin selmode
set sel [atomselect top "$selmode"]

puts -nonewline "\n Name output file? : "
gets stdin fname
set outf [open "$path/$fname" w]
set n [molinfo top get numframes]

	for {set i 0} {$i < $n} {incr i} {
	molinfo top set frame $i
	set ind_cont [$sel get index]
	#set ind_cont [lsort $ind]
	set num_sel [$sel num]
	
	incr num_sel -1
	for {set j 0} {$j < $num_sel} {incr j} {
		#set radgyr [measure rgyr $sel]
		
		set counter $j
		set counter2 $j
		incr counter2 1
		
		set ang_sel [lindex $ind_cont $counter]
		set ang_sel2 [lindex $ind_cont $counter2]
		
		#set radgyr [measure rgyr $sel]
		set c {}
		lappend c $ang_sel $ang_sel2
		#puts -nonewline "\n $c"
		set radgyr [measure bond $c]
		puts $outf "$radgyr"
	}
}
close $outf
puts -nonewline "\n $num_sel  "
puts "Done."
