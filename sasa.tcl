######################################
#                                    #
# DESCRIPTION:                       #
#       Compute SASA with vmd        #
#                                    #   
# USAGE:                             #
#         source sasa.tcl            #
#                                    #
######################################

set path [pwd]
set selmode "resname L13 L10 NP TPH" #Change selection to include all the residues
set probe 2.38 #Probe radius
set sel [atomselect top "$selmode"]
set n [molinfo top get numframes]
puts -nonewline "\n \t $n"
# Create/open file.dat to store the results
set output [open "$path/SASA_$selmode.dat" w] 
# sasa calculation loop
for {set i 0} {$i < $n} {incr i} {
	molinfo top set frame $i
	set sasa [measure sasa $probe $sel]
	puts "\t \t progress: $i/$n"
	puts $output "$sasa"
}
puts "\t \t progress: $n/$n"
puts "Done."	
close $output
