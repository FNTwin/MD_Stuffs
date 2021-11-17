#!/usr/bin/env python

import numpy as np
from argparse import ArgumentParser
import os
import sys
import os.path


parser = ArgumentParser()
groupT = parser.add_argument_group('INPUT OPTIONS', '')
groupT.add_argument("-i",
                    dest="traj",
                    type=str,
                    metavar="TRAJ",
                    help="Path to the nc file")
groupT.add_argument("-p",
                    dest="top",
                    type=str,
                    metavar="TOP",
                    help="Topology")
groupT.add_argument("-log",
                    dest="log",
                    metavar="OPTIONAL",
                    const=None,
                    nargs='?',
                    help="Path to the cleaned log file. If not given it will extract the windows from the trajectory")
groupO = parser.add_argument_group('PROCESSING OPTIONS', '')
groupO.add_argument("--space",
                    dest="dist_windows",
                    type=float,
                    metavar="1.0",
                    default=1.0,
                    help="Spacing between windows [Def 1.0 Angstrom]")
groupO.add_argument("--tol",
                    dest="tol",
                    type=float,
                    metavar="0.05",
                    default=0.05,
                    help="Tollerance [Def 0.05]")
groupO.add_argument("--target-res",
                    dest="target_res",
                    type=str,
                    default="TTT",
                    help="Target residue for COM-COM distance calculation. Only use this if --frame-miscount option is enabled [Default]")
groupO.add_argument("--clean",
                    dest="clean",
                    action="store_true",
                    default=False,
                    help="Automatically clean the log file")
groupO.add_argument("--frame-miscount",
                    dest="frame_miscount",
                    action="store_true",
                    default=False,
                    help="Use this option if the frames of the trajectory are different from the frame of the log files")
groupO.add_argument("--sh-file",
                    dest="sh_file",
                    action="store_true",
                    default=False,
                    help="Create a shell file to and explanations to produce the restrains and run the windows with the .rst file.")
groupO.add_argument("--denomination",
                    dest="denomination",
                    action="store_true",
                    default=False,
                    help="Normally the .rst file are written as frame_xxxx.rst where xxxx is the distance of the frame. With this option frames will be denominated as frame_1 .... This disable the --sh-file option")


args = parser.parse_args()
tol = args.tol
space=args.dist_windows

def cpptraj(args,n,tot_windows):
    #Write .in file with trajectory, the frame to extract and the restart file
    with open('extract.in','w') as f:
        f.write('trajin %s %d %d 1 \n' % (args.traj,n+1,n+1))
        if not args.denomination:
            f.write('trajout frame_%.3f.rst restart \n' % (tot_windows))
        else:
            f.write('trajout frame_%d.rst restart \n' % (tot_windows))
    #Execute file with cpptraj input mode from shell and remove extract.in and log.out file
    os.system('$AMBERHOME/bin/cpptraj %s < extract.in > out' % args.top)
    os.system('rm extract.in')
    os.system('rm out')

def select_frame(points,dist,tol):
    #Iterate between all the standard points at 1A of spacing and select the frame between the tollerance
    tot_windows=0
    dist_list=[]
    frame_list=[]
    for i in range(len(points)):
        counter=0
        for n,dist_frame in enumerate(dist):
            if (points[i]-tol)<dist_frame<(points[i]+tol) and counter<1:
                counter+=1
                tot_windows+=1
                print(f">>> Window {tot_windows:>3d} found at frame {n:>5d}: distance {dist_frame:>.3f} A")
                frame_list.append(n)
                dist_list.append(float(dist_frame))
                if not args.denomination:
                    cpptraj(args,n,dist_frame)
                else:
                    cpptraj(args,n,tot_windows)
    print(f">>> Total number of windows: {tot_windows:>3d}\n>>> Spacing of: {args.dist_windows:>.2f} A\n>>> Tollerance of: {tol:>.3f}")
    return dist_list, frame_list

def clean_log(args):
    with open(args.log,"r") as f:
        lines=f.readlines()
        formatted_text='{:>10} {:<20}\n'.format("#Frame", "Distance")
        for i,line in enumerate(lines):
            if i!=0:
                splitted_line=line.split()
                line_formatted="{:>10} {:<20}\n".format(splitted_line[0],splitted_line[7])
                formatted_text+=line_formatted

    filename="clean_"+os.path.basename(args.log)
    args.log=os.path.join(os.path.dirname(args.log),filename)
    with open(args.log,"w") as f:
        f.write(formatted_text)

if args.log is None:
    args.frame_miscount=True
else:
    if args.clean:
        clean_log(args)

#Check and set if AMBERHOME is a present environment to use cpptraj
amberhome = os.getenv('AMBERHOME')
if amberhome == None:
    print('Error: AMBERHOME not set')
    sys.exit(1)

if not args.frame_miscount:
    dist_frames, dist=np.loadtxt(args.log, skiprows=1, unpack=True)
    start,end=dist[0],dist[-1]
    
    if start < end: pass 
    else:
        space*=-1
    
    points=np.arange(start,end+space,space)
    dist_list, frame_list=select_frame(points,dist,tol)

else:
    print(f">>> Computation will be done on the trajectory, it could take a while!")
    import pytraj as pt
    from math import ceil
    traj=pt.iterload(args.traj, args.top)
    NP_dist=pt.calc_center_of_geometry(traj, ':NP')
    TTT_dist=pt.calc_center_of_geometry(traj, f':{args.target_res}')
    dist = np.linalg.norm(NP_dist - TTT_dist,axis=1) #compute all NP-TTT distance between frames

    start,end=dist[0],dist[-1] #IF PULLING IS DONE RIGHT SORTING IS NOT NEEDED
    if start < end: pass 
    else:
        space*=-1
    points=np.arange(start,end+space,space)
    dist_list,frame_list=select_frame(points,dist,tol)

with open("Recap_file.txt","w") as f:
    f.write("{:<12} {:<12} {:<12}\n".format("#WINDOW","FRAME", "DISTANCE"))
    dist_list_formatted=[]
    for i,(dist,frame) in enumerate(zip(dist_list, frame_list)):
        dist_list_formatted.append("{:.3f}".format(dist))
        f.write("{:<12} {:<12} {:<12.3f}\n".format(i+1, frame, dist))

if args.sh_file and not args.denomination:

    helper_comment="""#!/bin/bash

#This basic scripts will iterate between the distances and change the restr_no_pull.txt to the right values
#To run this .sh script you need to change the r2 and r3 value of your distance restraint to WINDOWFLAG in the restr_no_pull.txt
#It should be ... r1=99.0, r2=WINDOWFLAG, r3=WINDOWFLAG, r4=99.0 ...
#This is just a standard loop to create a dir and change the flag value to the correct one
    """
    loop="""
for distance in %s ; do
mkdir dist_${distance}
cd dist_${distance}
cp ../restr_no_pull.txt .
sed -i "s/WINDOWFLAG/${distance}/g" restr_no_pull.txt

#YOUR AMBER THING
#pmemd.cuda -O -i ../mdin.in \\
#              -o window_${distance}.out \\
#              -p your_prmtop \\
#              -r window_${distance}.rst \\
#              -c ../frame_${distance}.rst \\
#              -x window_${distance}.nc 

cd ../

done
    """ % " ".join(dist_list_formatted) #(" ".join(list(map(lambda x : str(round(x,3)), dist_list))))
    with open("windows_helper.sh", "w") as f:
        f.write(helper_comment)
        f.write(loop)
