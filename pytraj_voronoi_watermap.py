#!/usr/bin/env python3

import os
from argparse import ArgumentParser
from os import mkdir
from os.path import isdir
import numpy as np
import matplotlib.pyplot as plt
import pytraj as pt
from matplotlib import colors,cm
from matplotlib.colors import Normalize
from numpy import linspace, array, loadtxt, meshgrid, isnan, where, concatenate, pi
from numpy import amin, amax
from scipy.interpolate import griddata, Rbf
from scipy.spatial import Voronoi, voronoi_plot_2d
from MyUtil.ChemObj import parse_pdb_lines_list, Atom
from MyUtil.var import pytraj_load_traj

#TODO: Implementare un metodo per importare un json con configurazioni di matplotlib e un modo per poter
# plottare le watermaps sulla tassellazione di voronoi 
parser = ArgumentParser()

groupT = parser.add_argument_group('OPTIONS TO SELECT INPUT TRAJ AND PRMTOP', '')
groupT.add_argument("--traj-dir",
                    dest="traj_dir",
                    default="./prod/",
                    type=str,
                    metavar="PATH",
                    help="path to the directory containing the input trajectories [./prod/]")
groupT.add_argument("--traj-prefix",
                    dest="traj_prefix",
                    default="NPT_MC_",
                    type=str,
                    metavar="BEG",
                    help="prefix of the input trajectories [NPT_MC_]")
groupT.add_argument("--traj-suffix",
                    dest="traj_suffix",
                    default="noWAT.nc",
                    type=str,
                    metavar=".nc",
                    help="suffix of the input trajectories [noWAT.nc]")
groupT.add_argument("--prmtop",
                    dest="prmtop",
                    default="noWAT*.prmtop",
                    type=str,
                    help="prmtop file [noWAT*.prmtop]")
groupT.add_argument("--stride",
                    dest="stride",
                    default=1,
                    type=int,
                    help="keep every 'stride' frame from trajectory [1]")
groupT.add_argument("--start",
                    dest="start",
                    default=0,
                    type=int,
                    help="first frame to read [0]")
groupT.add_argument("--stop",
                    dest="stop",
                    default=0,
                    type=int,
                    help="last frame to read [0]")
groupT.add_argument("--skip",
                    dest="skip",
                    default=0,
                    type=int,
                    help="number of frame to skip [0] (to use only with --start and --stop, otherwise use --stride")
groupT.add_argument("--gro-traj",
                    dest="gro_traj",
                    default="",
                    help="input gromacs xtc trajectory, need also the .top file")
groupT.add_argument("-s",
                    dest="script",
                    default=False,
                    action="store_true",
                    help="running in a script, don't ask anything!")
groupT.add_argument("-u",
                    dest="use",
                    default=False,
                    action="store_true",
                    help="If a saved file is found, it's content is loaded instead of calculate with pytraj"
                         " (must be created from the same commands!!!)")

groupR = parser.add_argument_group('OPTIONS TO COMPUTE THE VORONOI TASSELLATION', '')
groupR.add_argument("--ligand",
                    dest="ligand",
                    #nargs='+',
                    type=str,
                    help="the resname of the ligands [--ligand L10,L13]")
groupR.add_argument("--com",
                    dest="com",
                    type=str,
                    nargs="+",
                    default=None,
                    help="Lists of atoms names for the center of mass calculation." +
                     "Examples: [--ligand 'L1,L2' --com 'C6,C7,C8,N2' 'C3,C4'] L1 will use C6,C7,C8,N2 and L2 will use C3,C4 [--ligand 'L1,L2' --com 'C3,C4'] both ligands will use C3,C4 " +
                    "[--ligand 'L1,L2'] will use the entire ligands to calculate the COM")
groupR.add_argument("--progression",
                    dest="progression",
                    type=int,
                    default=None,
                    help="Input how many times I should calculate the voronoi tassellation. Trajectory will be divided into N averaged frames and N plots will be produced")
groupR.add_argument("--treshold",
                    dest="threshold",
                    type=float,
                    default=0.4,
                    help="Area threshold. Standard is 0.4.")
groupR.add_argument("--watermap",
                    dest="watermap",
                    action="store_true",
                    default=False,
                    help="Flag to activate watermap plotting on top of the voronoi tassellation. [Def. Fault]")
groupR.add_argument("--watermap-path",
                    dest="watermap_path",
                    default=str,
                    nargs="+",
                    help="List of path to the watermap .txt data (Usually in analysis/raw_data/)[Ex. path1 path2 ...]")


opt = parser.parse_args()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class Trajectory_NP:
    def __init__(self):
        self.trajectory = None
        #pytraj_load_traj(beg=opt.traj_prefix, end=opt.traj_suffix, prmtop=opt.prmtop, iter_load=True,
         #                     search_dir=opt.traj_dir, stride=opt.stride, start=opt.start, stop=opt.stop,
          #                    skip=opt.skip)
        #print(">>> Computing the %s surface with %s as ligand" % (opt.solvent, opt.ligand))


class Residue(object):
    def __init__(self, mask=None):
        self.atoms = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.res_num = 1
        self.mask=mask

    def add_atom(self, atom_to_add):
        self.atoms.append(atom_to_add)
     
    def res_name(self):
        self._res_name=self.atoms[0].res_name
        return self._res_name
       
    @staticmethod
    def calculate_COM(array_of_positions):
        return np.sum(array_of_positions,axis=0)/array_of_positions.shape[0]
    
    def COM(self):
        self.com_mask=self.mask[self.res_name()]
        try:
            selected_atoms=list(filter(lambda x: x.atom_name in self.com_mask.split(","), self.atoms))
        except AttributeError as error:
            selected_atoms=self.atoms
        tmp_atom_array=[]
        for atom in selected_atoms:
            tmp_atom_array.append([atom.x,atom.y,atom.z])
        X_com,Y_com,Z_com = self.calculate_COM(np.array(tmp_atom_array))
        self._COM=Atom(x=X_com, y=Y_com, z=Z_com)
        
    def unitVec(self):
        relative_sys_orig_atom = self.atoms[0]
        X_o,Y_o,Z_o = relative_sys_orig_atom.x,relative_sys_orig_atom.y,relative_sys_orig_atom.z
        if not hasattr(self, "_COM"):
            self.COM()
        X_com, Y_com, Z_com = self._COM.x, self._COM.y, self._COM.z           
        X_v, Y_v, Z_v = X_com-X_o, Y_com - Y_o, Z_com - Z_o
        com_vector=[X_v, Y_v, Z_v]
        self._unitVec = [i/(X_v**2 + Y_v **2 + Z_v**2)**.5 for i in com_vector]
        return self._unitVec
    
    def spatial_data(self, mask=None):
        if not hasattr(self, "_COM"):
            self.COM()
        
        return [self._COM.x,self._COM.y,self._COM.z,*self.unitVec()]
    
    def all_data(self, mask=None):
        if not hasattr(self, "_COM"):
            self.COM()
        return [self._COM.x,self._COM.y,self._COM.z,*self.unitVec(), self.res_name()]


class Voronoi_NP_frame():
    from scipy.spatial import Voronoi, voronoi_plot_2d

    def __init__(self, atoms_list, atom_dict_mask=None):
        self.residue_list = create_residue_list(atoms_list, atom_mask=atom_dict_mask)

    def calculate_info(self):
        self.db_no_dist_matrix, self.dist_matrix, self.sph_coord, self.res_dict = create_dataset(self.residue_list)

    def do_voronoi(self):
        #self.sph_coord[:,1]=np.cos(self.sph_coord[:,1])

        self.vor = Voronoi(self.sph_coord)

    def compute_area(self):
        fact=opt.threshold
        #fact=0.4 #STANDARD
        area_poly=[]
        polygon_list=[]
        for r in range(len(self.vor.point_region)):
            region = self.vor.regions[self.vor.point_region[r]]
            if not -1 in region:
                polygon = [self.vor.vertices[i] for i in region]
                tmp_area=Area(polygon)
                if tmp_area < fact:
                    polygon_list.append(polygon)
                    area_poly.append(Area(polygon))
        self.polygons, self.polygons_area =polygon_list, area_poly
        self.minima = min(self.polygons_area)
        self.maxima = max(self.polygons_area)

    def renormalize(self,minima,maxima):
        fig= plt.figure(figsize=(14, 10),tight_layout=True)
        ax = plt.subplot(111)
        data=self.sph_coord
        plt.rcParams['font.size'] = 34
        #line_colors="darkslategray"
        voronoi_plot_2d(self.vor, ax=ax, show_points=False, show_vertices=False, line_colors="white", line_width=3)
        step=0.1
        plt.xlim(float(np.min(data[:,0])) - step ,float(np.max(data[:,0])) + step)
        plt.ylim(float(np.min(data[:,1])) - step ,float(np.max(data[:,1])) + step)
        self.minima = minima
        self.maxima = maxima
        norm = MidpointNormalize(vmin=self.minima, vmax=self.maxima, midpoint=(self.maxima-self.minima)/2.1, clip=True) 
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm_r)
        for i in range(len(self.polygons)):
            tmp_pol=np.vstack(self.polygons[i])
            plt.fill(*tmp_pol.T, color=mapper.to_rgba(self.polygons_area[i]))

        self.ax=ax
        self.plt=plt


    def plot(self):
        fig= plt.figure(figsize=(14, 10),tight_layout=True)
        ax = plt.subplot(111)
        plt.rcParams['font.size'] = 34
        data=self.sph_coord
    
        #Stuff for plotting
        #plt.xticks(color='w')
        #plt.yticks(color='w')
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(3)
        plt.tick_params('x', width=3, length=14)
        plt.tick_params('y', width=3, length=14)
    
        #fig, ax = plt.subplots(figsize=(14, 10))

        #plt.xlim(float(np.min(data[:,0])) + step ,float(np.max(data[:,0])) + step)
        #plt.ylim(float(np.min(data[:,1])) + step ,float(np.max(data[:,1])) + step)

        #NORMALIZATION 
        #Colorate the voronoi area based of the area value 
        voronoi_plot_2d(self.vor, ax=ax, show_points=False, show_vertices=False,
            line_colors="darkslategray", line_width=3)    
        plt.xlabel("φ")
        plt.ylabel("cos(θ)") 
        plt.xlim(-3,3)
        plt.ylim(-1,1)
        norm = MidpointNormalize(vmin=self.minima, vmax=self.maxima, midpoint=(self.maxima-self.minima)/2.1, clip=True) 
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm_r)
        for i in range(len(self.polygons)):
            tmp_pol=np.vstack(self.polygons[i])
            plt.fill(*tmp_pol.T, color=mapper.to_rgba(self.polygons_area[i]), alpha= 0.7)
        ax.scatter(data[:,0], data[:,1], color="darkslategray")
        plt.savefig("./analysis/%s_nowatermap.pdf" % basetitle)
        plt.savefig("./analysis/%s_nowatermap.jpeg" % basetitle)   
        plt.cla()  

        if opt.watermap:
            voronoi_plot_2d(self.vor, ax=ax, show_points=False, show_vertices=False,
                line_colors="white", line_width=3)
            plt.xlabel("φ")
            plt.ylabel("cos(θ)") 
            plt.xlim(-3,3)
            plt.ylim(-1,1)
            t,c,v=(np.array([5,7,8])-1)
            for wtrm in opt.watermap_path:
                arr=np.loadtxt(wtrm)
                theta, phi, valore = arr[:,t], arr[:,c], arr[:,v]
                countour=plot_watermap(theta, phi, valore)
                plt.scatter(data[:,0], data[:,1], color="white")
                #plt.savefig("./analysis/%s.%s.pdf" % (basetitle, os.path.basename(wtrm)[:-4]))
                plt.savefig("./analysis/%s.%s.jpeg" % (basetitle, os.path.basename(wtrm)[:-4]))
                #iterate the countourf plot collection and remove them from the figure
                #so the next plot will not be on top of each others (optimization...it works either way)
                for countour in countour.collections:
                    countour.remove()

        self.ax=ax
        self.plt=plt
    
    def do_what_you_have_to_do(self):
        self.calculate_info()
        self.do_voronoi()
        self.compute_area()
        #self.plot()
        return self

def plot_watermap(phi, costheta, valore):
    x, y, z = phi, costheta, valore
    #-np.pi/2
    n_countourf = 400
    xi = linspace(-3, 3, 50)
    yi = linspace(-1, 1, 50)
    #yi = linspace(-3, 3, 50)
    X, Y = meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='cubic', rescale="true")
    rbf3 = Rbf(x, y, z, function='cubic', smooth=0)
    Z_rbf = rbf3(X, Y)

    # Extrapolation using a RBF with low smoothness
    mask = isnan(Z)
    z_min, z_max = min(valore), max(valore)
    idx = where(~mask, Z, Z_rbf)
    idx[idx < z_min] = 0
    idx[idx > z_max] = z_max

    #cmap="Greys_r"
    countour=plt.contourf(X, Y, idx, n_countourf, cmap="Blues",
                 norm=Normalize(vmin=0., vmax=np.max(idx)*1.4))

    plt.rcParams['font.size'] = 34
    plt.tick_params('x', width=3, length=14)
    plt.tick_params('y', width=3, length=14)
    return countour





def create_residue_list(atoms_list, atom_mask=None):
    res_numbers=list(set(map(lambda x: x.res_num, atoms_list)))
    red_list=[]
    tmp_res=Residue(mask=atom_mask)
    old=min(res_numbers)
    for i in range(0,len(atoms_list)):
        if atoms_list[i].res_num != old or i==len(atoms_list)-1:
            old=atoms_list[i].res_num
            red_list.append(tmp_res)
            tmp_res=Residue(mask=atom_mask)
        tmp_res.add_atom(atoms_list[i])
    return red_list

def create_dataset(residue_list):
    res_set=set(map(lambda x: x.res_name(), residue_list))
    n_residues=len(res_set)
    res_dict={i:[] for i in res_set}
    tmp_data=[]
    spherical=[]
    COM_list=[]
    COM_list_dist=[]
    for i in residue_list:
        tmp_data.append(i.spatial_data(atom_mask))
        i.COM()
        COM_list.append(i._COM)
        COM_list_dist.append([i._COM.x,i._COM.y,i._COM.z])
        i._COM.add_spherical_coordinates(ISO=True, mathematics=False, from_zenith=True)
        spherical.append([i._COM.phi,np.cos(i._COM.theta)])
        if i.res_name() in res_dict:
            res_dict[i.res_name()].append([i._COM.phi,np.cos(i._COM.theta)])
    dist_matrix=calc_dist(COM_list_dist)
    db_no_dist_matrix=np.array(tmp_data)
    sph_coord=np.array(spherical)
    return db_no_dist_matrix, dist_matrix, sph_coord, res_dict

def calc_dist(mat):
    from sklearn.metrics import pairwise_distances
    return pairwise_distances(mat)

def cos_sim(mat):
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(mat)

def Area(points):
        #Points are polyhedra vertex
        #Use the shoelace formula to compute the area
        tmp=np.vstack(points)
        x,y=tmp[:,0],tmp[:,1]
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def load_files(args):
    files = []
    for i, input_path in enumerate(args.inp):
        if os.path.isdir(input_path):
            for file in os.listdir(input_path):
                if file.endswith(".txt", ".pdb"):
                    files.append(os.path.join(input_path, file))
        else:
            files.append(input_path)
    return files


class Current_job():
    system=[]
    minima=None
    maxima=None
    def __init__(self, NP=None, minima=None, maxima=None):
        if NP:
            self.system.append(NP)
            self.check_min_max(NP)
    
    def check_min_max(self, NP):
        if (self.minima and self.maxima) is None:
            self.minima,self.maxima=NP.minima,NP.maxima
        else:
            self.minima=min(self.minima,NP.minima)
            self.maxima=max(self.maxima,NP.maxima)

    def add_NP(self, NP):
        NP.do_what_you_have_to_do()
        self.__init__(NP, NP.minima, NP.maxima)

    def plot_system(self):
        print(len(self.system))
        for NP in self.system:
            NP.renormalize(self.minima,self.maxima)
            NP.plt.show()

class IterNP():

    def __init__(self):
        self.system=[]
        self.n=0

    def __next__(self):
        if self.n < len(self.system):
            tmp=self.system[self.n]
            self.n+=1
            return tmp
        else:
            self.n=0
            raise StopIteration

    def add(self, NP):
        self.system.append(NP)


    def __iter__(self):
        return self

    def __len__(self):
        return len(self.system)



class Voronoi_NP(Voronoi_NP_frame):
    def __init__(self):
        self.iter=IterNP()

    def add_NP(self, NP):
        #self.system.append(NP)
        self.iter.add(NP)
        self.residue_list=NP.residue_list


    def calculate_info(self):
        tmp=0
        for NP in self:
            NP.calculate_info()
            tmp+= NP.sph_coord
        self.sph_coord=tmp/len(self)


    def __iter__(self):
        return self.iter

    def __len__(self):
        return len(self.iter)

def validate_atom_mask(ligand_mask, atom_mask):
    tmp_lig_mask=ligand_mask.replace(":","").split(",")
    tmp_dict={}
    print("Center of Mass calculation on the following masks:")
    if atom_mask is None:
        for i in tmp_lig_mask:
            tmp_dict[i]=None
        return tmp_dict
    assert len(atom_mask) <= len(tmp_lig_mask), f"{bcolors.FAIL} Number of atoms mask is greater than number of ligands. Script stopped. {bcolors.ENDC}"
    if len(tmp_lig_mask) == len(atom_mask):
        for n,i in enumerate(tmp_lig_mask):
            tmp_dict[i]=atom_mask[n]
            print(i , " : ",bcolors.HEADER,atom_mask[n],bcolors.ENDC)
    elif len(atom_mask) == 1:
        for i in tmp_lig_mask:
            tmp_dict[i]=atom_mask[0]
            print(i , " : ",bcolors.HEADER,atom_mask[0],bcolors.ENDC)
    return tmp_dict

    

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if not isdir("analysis"): mkdir("analysis")
if not isdir("analysis/raw_data"): mkdir("analysis/raw_data")
if not isdir("analysis/acorr"): mkdir("analysis/acorr")


trajectory=pytraj_load_traj(beg=opt.traj_prefix, end=opt.traj_suffix, prmtop=opt.prmtop, iter_load=True,
                              search_dir=opt.traj_dir, stride=opt.stride, start=opt.start, stop=opt.stop,
                              skip=opt.skip)
centered_trajectory = pt.center(trajectory, ":NP", center="origin")
rms_fitted_traj = pt.align(centered_trajectory, ref=0, mask=":NP")
stripped_HandS_traj = rms_fitted_traj.strip(mask="@H= | @SH | @S | :Na+,Cl-")
#traj=rms_fitted_traj.strip(mask=":NP")

top = pt.load_topology(opt.prmtop)
ligand_mask=":%s" % opt.ligand
print(f"Voronoi tassellation calculated on the {bcolors.OKGREEN} {ligand_mask} {bcolors.ENDC} ligand mask")

atom_mask=opt.com
indexes=top.select(ligand_mask)
mask_dict=validate_atom_mask(ligand_mask,atom_mask)

nano=Voronoi_NP()

if opt.progression is not None:
    tmp_traj=[]
    n_frames=len(trajectory)
    tmp=int(n_frames / opt.progression)
    start=0
    end=tmp
    for i in range(opt.progression):
        print(start,end)
        n_fr=np.arange(start, end, 1)
        print(n_fr)
        tmp_traj.append(pt.get_average_frame(rms_fitted_traj, mask=ligand_mask, frame_indices=n_fr, dtype="traj", autoimage=False, rmsfit=None, top=None))
        start=end
        end+=tmp
else:
    n_fr=None
    tmp_traj=pt.get_average_frame(rms_fitted_traj, mask=ligand_mask, frame_indices=n_fr, dtype="traj", autoimage=False, rmsfit=None, top=None)

trajectory2=tmp_traj
for k,i in enumerate(tmp_traj):
    if isinstance(i, pt.trajectory.trajectory.Trajectory):
        i=i[0]
    print(i)
    tmp_list=[0] * len(indexes)
    for l,j in enumerate(i): 
        atom_type=list(trajectory.top.atoms)[indexes[l]]
        tmp_list[l]=Atom(x=j[0],y=j[1],z=j[2], res_name=atom_type.resname, atom_name=atom_type.name , res_num=atom_type.resid)

    test=Voronoi_NP_frame(tmp_list, atom_dict_mask=mask_dict)
    nano.add_NP(test)
    basetitle="ligands_Voronoi_%s.%d" %(atom_mask,k)
    nano.do_what_you_have_to_do()
    nano.plot()

nano.do_what_you_have_to_do()
nano.plot()

