set path [pwd]
puts -nonewline "\n Select the polymer (e.g fragment 1): "
gets stdin selmode
set sel [atomselect top "$selmode"]
#set sel [atomselect top "type Au"]
puts -nonewline "\n Name output file? : "
gets stdin fname
set outf [open "$path/$fname" w]
set n [molinfo top get numframes]


########
set arr(0) 0
for {set i 1} {$i <= 181} {incr i} {
    set arr($i) 0
}
########

for {set i 0} {$i < $n} {incr i} {
	molinfo top set frame $i
	set ind_cont [$sel get index]
	#set ind_cont [lsort $ind]
	set num_sel [$sel num]
	
	incr num_sel -2
	for {set j 0} {$j < $num_sel} {incr j} {
		#set radgyr [measure rgyr $sel]
		
		set counter $j
		set counter2 $j
		incr counter2 1
		set counter3 $counter2
		incr counter3 1
		
		set ang_sel [lindex $ind_cont $counter]
		set ang_sel2 [lindex $ind_cont $counter2]
		set ang_sel3 [lindex $ind_cont $counter3]
		
		set c {}
		lappend c $ang_sel $ang_sel2 $ang_sel3
		
		set radgyr [measure angle $c]

		set index_ang [expr int($radgyr)]
		incr arr($index_ang) 1

		#puts $outf "$radgyr"
	}
	#puts $outf "FRAME"
}
for {set i 0} {$i <= 181} {incr i} {
    puts $outf "$arr($i)"
}
close $outf
puts -nonewline "\n $num_sel  "
puts "Done."
