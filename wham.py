#!/usr/bin/env amber.python

import numpy as np
from argparse import ArgumentParser
import os, sys, glob
import matplotlib.pyplot as plt
import pytraj as pt
from typing import List,Union,Tuple
from pathlib import Path

parser = ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("-i",
                    dest="input",
                    type=str,
                    metavar="TRAJ NAME",
                    help="Name of the trajectory, for each window the name must be the same!")
group.add_argument("-l",
                    dest="log",
                    type=str,
                    metavar="LOG",
                    default="window.dat",
                    help="Log file name option. Processing will be done on the distance log file [Use window.dat]")
groupT = parser.add_argument_group('INPUT OPTIONS', '')
groupT.add_argument("-p",
                    dest="top",
                    type=str,
                    nargs='?',
                    metavar="TOPOLOGY",
                    help="Path or topology file, only if -i is used [Optional]")
groupT.add_argument("--dir",
                    dest="traj_dir",
                    default="prod/",
                    type=str,
                    metavar="DIR",
                    help="Folder name containing the input trajectories or the log file [Def prod/]")
groupT.add_argument("--target",
                    dest="res",
                    type=str,
                    metavar="TARGET RES",
                    default="TTT",
                    help="Target residue [Default TTT]")
groupD = parser.add_argument_group('PROCESSING OPTIONS', '')
groupD.add_argument("--plot",
                    dest="plot",
                    action="store_true",
                    help="Plot the distance data and the calculated histograms")
groupD.add_argument("--force",
                    dest="force",
                    type=float,
                    metavar="5.0",
                    default=5.0,
                    help="Restraint force of amber * 2 as defined for wham [Def 5.0]")
groupD.add_argument("--process-logs",
                    dest="proc_log",
                    action="store_true",
                    help="Process the logs into one unique file named window.dat")
groupW = parser.add_argument_group('WHAM OPTIONS', '')
groupW.add_argument("--wham",
                    dest="wham",
                    action="store_true",
                    help="Try to execute wham! A default command will be displayed even without this option!")
groupW.add_argument("--temp",
                    dest="temp",
                    metavar="298",
                    default=298,
                    type=float,
                    help="Temperature for the PMF calculation [Def 298]")
groupW.add_argument("--n-bins",
                    dest="n_bins",
                    type=int,
                    metavar="10",
                    default=10,
                    help="Number of bins for the PMF calculation [Def 10/window]")
groupW.add_argument("--tol",
                    dest="tol",
                    type=float,
                    metavar="0.00000001",
                    default=0.00000001,
                    help="WHAM tollerance [Def 0.00000001]")

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    OKCYAN = '\033[96m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def write_wham_file(path : Union[str, Path],
                    dist : float,
                    force : float) -> None: 
    dist_float=dist.split("_")[1]
    with open("wham_metadata.dat", "a") as wh:
        wh.write("{}\t{}\t{:.3f}\n".format(path,dist_float,force))

def write_dist_file(distances : Union[List[float] ,np.ndarray],
                    path : Union[str, Path], 
                    custom_frames : Union[List[float] ,np.ndarray] = None) -> None:
    with open(path+"/distances.dat", "w") as wh2:
        if custom_frames is None:
            for i,distance in enumerate(distances,1):
                wh2.write("{:<10d}\t{:<20.3f}\n".format(i,distance))
        else:
            for i,distance in zip(custom_frames,distances):
                wh2.write("{:<10d}\t{:<20.3f}\n".format(int(i),distance))

def normal_PDF(bins : Union[List[float] ,np.ndarray], mu : float , sigma : float) -> np.ndarray:
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2))

def create_log_file(path, distances):
    frames=np.arange(0,len(distances),1)
    with open(os.path.join(path,"window.dat"), "w") as wh:
        wh.write('{:>10} {:<20}\n'.format("#Frame", "Distance"))
        for frame, distance in zip(frames,distances):
            wh.write('{:>10} {:<20.3f}\n'.format(frame+1, distance))

def join_logs(path):
    all_distances=[]
    for file in glob.glob(os.path.join(path,'*.dat')):
        if os.path.basename(file) == "window.dat" or os.path.basename(file) == "distances.dat":
            pass
        else:
            _, distances=np.loadtxt(file, skiprows=1, unpack=True)
            all_distances.extend(distances)

    create_log_file(path, all_distances)
    

def prepare_paths(args):
    main_path=os.getcwd()
    search_dir=os.path.join(main_path,args.traj_dir)
    #frame_directs= next(os.walk('.'))[1]
    frame_directs=next(os.walk(search_dir))[1]
    correct_frames=[]
    full_paths=[]
    for i in frame_directs:
        try:
            #name=os.path.join(main_path,i,args.traj_dir,choice)
            name=os.path.join(main_path,args.traj_dir,i,choice)
            if args.proc_log:
                #join_logs(os.path.join(main_path,i))
                join_logs(os.path.join(main_path,args.traj_dir,i))
            if os.path.isfile(name):
                full_paths.append(name)
                correct_frames.append(i)
            else:
                print(bcolors.WARNING,"File not found in %s" % i, bcolors.ENDC)
        except TypeError as error: 
            print(error)
            
    return {"traj":full_paths, "frames": frame_directs}

def calc_distances(traj : pt.Trajectory ,res : str) -> np.ndarray:
    NP_dist=pt.calc_center_of_geometry(traj, ':NP')
    RES_dist=pt.calc_center_of_geometry(traj, res)
    return np.linalg.norm(NP_dist - RES_dist,axis=1)

def plot(distances :  Union[List[float] ,np.ndarray],
         path : Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    mu= abs(distances.mean())
    sigma=distances.std()
    fig,axs=plt.subplots(ncols=2, gridspec_kw={'width_ratios': [5, 1]},figsize=(15, 8), sharey=True)
    axs[0].plot(np.arange(0,distances.shape[0],1),distances, color="black", linewidth=0.4)
    count, bins, _ =axs[1].hist(distances, bins=20, orientation="horizontal", alpha=0.3, density=True)
    #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.subplots_adjust(bottom=0.15, wspace=0)
    axs[0].axhline(distances.mean(), color="red", linestyle="--")
    axs[1].axhline(distances.mean(), color="red", linestyle="--", label=f"{distances.mean():.3f}")
    axs[1].yaxis.tick_right()
    axs[1].tick_params(right=False)
    axs[0].title.set_text("Trajectory distance")
    axs[1].title.set_text("Distance histogram")
    plt.plot(normal_PDF(bins,mu,sigma),bins,linewidth=2, color='r')
    axs[1].legend()
    plt.savefig(os.path.join(path,"hist.jpeg"))
    count, bins, _ =axs[1].hist(distances, bins=20, orientation="horizontal", alpha=0.3, density=False)
    plt.close()
    return count,bins

def plot_all(hists : List[Tuple[np.ndarray, np.ndarray]]) -> None:
    plt.figure(figsize=(15,8))
    for count,bins in hists:
        plt.plot(bins[:-1],count)
    plt.savefig(os.path.join(os.getcwd(),"hist_all.png"))
    plt.close()

def plot_wham():
    with open("wham_output.txt", "r") as file:
        lines= file.readlines()
        beg=1
        for n,line in enumerate(lines,1):
            if line.startswith("#Window"):
                end=n-1
        arr=np.zeros((end-1, 2))
        k=0
        for line in range(beg,end):
            dist, energy, _ , _ ,_ = lines[line].split()
            arr[k]=float(dist),float(energy)
            k+=1
        plt.style.use('seaborn-poster')
        plt.plot(*arr.T, linewidth=0.7, color="black")
        plt.xlabel("Distance [Ang]")
        plt.ylabel("Energy")
        plt.title("PMF - Wham")
        plt.savefig("wham_output.png")
        plt.close()


args = parser.parse_args()

if os.path.isfile(os.path.join(os.getcwd(),"wham_metadata.dat")):
    os.remove(os.path.join(os.getcwd(),"wham_metadata.dat"))

if args.input is None:
    print(bcolors.OKCYAN,">>> Log file will be used",bcolors.ENDC)
    choice=args.log
else:
    print(bcolors.OKCYAN,">>> Trajectory will be used, be sure to pass the topology",bcolors.ENDC)
    choice=args.input

res=":"+args.res

paths=prepare_paths(args)
hists=[]

for path,frame in zip(paths["traj"],paths["frames"]):
    frame_path=os.path.join(os.getcwd(),args.traj_dir,frame)
    if args.input is not None:
        traj=pt.load(path,args.top)
        distances=calc_distances(traj,res)
        write_dist_file(distances,frame_path)
    else:
        #Use clean_log.py for the correct format
        frame_numbers,distances=np.loadtxt(path, skiprows=1, unpack=True) #should rewrite the path for writewham
        write_dist_file(distances,frame_path, custom_frames=frame_numbers)
    if args.plot:
        hist=plot(distances,frame_path)
        hists.append(hist)
    write_wham_file(os.path.join(frame_path,"distances.dat"), frame, args.force)
if args.plot:
    plot_all(hists)

if not os.path.exists("wham_metadata.dat"):
    print(bcolors.WARNING,">>> wham_metadata.dat not found, something went wrong! Check your settings or your directories!",bcolors.ENDC)
    sys.exit(f"{bcolors.FAIL}TERMINATED")

frame_float=list(map(lambda x : float(x.split("_")[1]),paths["frames"]))
print(" >>> To execute wham you must do (or try the --wham option):")
print(" >>> wham min max n_bins tol T numpad metadatafile outfile")
print(" >>> By filling what I could, the command should probably be:")
print(bcolors.OKCYAN,">>> wham {:<6.3f} {:<6.3f} {:<3d} {} {:<5.2f} 0 wham_metadata.dat wham_output.txt".format(np.min(frame_float),
                                                 np.max(frame_float) , args.n_bins*len(frame_float),
                                                  str(args.tol), args.temp), bcolors.ENDC)

if args.wham:
    ex="wham {:<6.3f} {:<6.3f} {:<3d} {} {:<5.2f} 0 wham_metadata.dat wham_output.txt".format(np.min(frame_float), np.max(frame_float) , 
                                args.n_bins*len(frame_float), str(args.tol), args.temp)
    try:
        os.system(ex)
        plot_wham()
    except:
        print(bcolors.WARNING," >>> There was an error! Check that wham is set or try to use it yourself with the commands on top!")






