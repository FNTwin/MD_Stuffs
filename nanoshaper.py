#!/usr/bin/env python3

import os, argparse
import numpy as np
import pytraj as pt
from templates import ambersiz,prm_file
import subprocess
import shlex
import logging
import re
import matplotlib.pyplot as plt

logname="Nanoshaper_python.log"

try:
    os.remove(logname)
except FileNotFoundError as error:
    pass
    
logging.basicConfig(filename=logname, level=logging.INFO)

parser = argparse.ArgumentParser()

groupD = parser.add_argument_group('COMMON OPTIONS', '')
groupD.add_argument("-i",
                    dest="traj",
                    metavar="TRAJ",
                    type=str,
                    help="Trajectory path")
groupD.add_argument("-p",
                    dest="prmtop",
                    metavar="TOPOLOGY",
                    type=str,
                    help="Topology path")
groupD.add_argument("--stride",
                    dest="stride",
                    default=1,
                    metavar=1,
                    type=int,
                    help="Keep every 'stride' frame from trajectory [1]")


groupO = parser.add_argument_group('COMMON OPTIONS', '')


args = parser.parse_args()



#traj=pt.iterload(args.traj,args.prmtop,args.stride, frame_slice=range(0,len(traj),args.slice))

def cpptraj(args,frame):
    #Write .in file with trajectory, the frame to extract and the restart file
    with open('extract.in','w') as f:
        f.write('trajin %s %d %d 1 \n' % (args.traj,frame+1,frame+1))
        f.write('trajout tmp_frame.pdb\n')
    #Execute file with cpptraj input mode from shell and remove extract.in and log.out file
    os.system('$AMBERHOME/bin/cpptraj %s < extract.in > out' % args.prmtop)
    os.system('rm extract.in')
    os.system('rm out')

class Atom(object):
    def __init__(self, labels=None, pdb_line='', res_name='', atom_name='', atom_type='', element='', charge=0.0,
                 res_num=1, atom_num=1, x=0.0, y=0.0, z=0.0, kind='', zhi=None, chain_id='',
                 pytraj_atom=None, frame="none", traj="none", also_xyz=False):
        self.kind = kind
        self.atom_num = atom_num
        self.atom_name = atom_name
        self.altloc = ''
        self.res_name = res_name
        self.chainID = chain_id
        self.res_num = res_num
        self.x = x
        self.y = y
        self.z = z
        self.occupancy = 0.0
        self.tempfactor = 0.0
        self.element = element
        self.atom_type = atom_type
        self.charge = charge
        self.pdb_line = pdb_line
        self.np_coords, self.mycoords = None, None
        if labels: self.labels = labels
        else: self.labels = []
        if pdb_line: self.pdb_line_parser(in_line=pdb_line, zhi=zhi)
        if pytraj_atom: self.import_pytraj_atom(pytraj_atom=pytraj_atom, frame=frame,  traj=traj, also_xyz=also_xyz)
        self.distances = {}
        self.bonds = []
        self.neighbours = []

    def traslate(self, x=False, y=False, z=False):
        if x: self.x += x
        if y: self.y += y
        if z: self.z += z

    def scale(self, x=False, y=False, z=False):
        if x: self.x *= x
        if y: self.y *= y
        if z: self.z *= z

    def pdb_line_parser(self, in_line, reference=False, standard=True, return_labels=False, prev_at_idx=0, zhi=None):
        from numpy import array

        def add_with_label(items_list):
            for idx, value in enumerate(items_list):
                if self.labels[idx].lower() == 'kind': self.kind = value
                elif self.labels[idx].lower() == 'atom_num': self.atom_num = int(value)
                elif self.labels[idx].lower() == 'atom_name': self.atom_name = value
                elif self.labels[idx].lower() == 'altloc': self.altloc = value
                elif self.labels[idx].lower() == 'res_name': self.res_name = value
                elif self.labels[idx].lower() == 'chainid': self.chainID = value
                elif self.labels[idx].lower() == 'res_num': self.res_num = int(value)
                elif self.labels[idx].lower() == 'x': self.x = float(value)
                elif self.labels[idx].lower() == 'y': self.y = float(value)
                elif self.labels[idx].lower() == 'z': self.z = float(value)
                elif self.labels[idx].lower() == 'occupancy': self.occupancy = float(value)
                elif self.labels[idx].lower() == 'tempfactor': self.tempfactor = float(value)
                elif self.labels[idx].lower() == 'element': self.element = value
                elif self.labels[idx].lower() == 'charge': self.charge = float(value)
        # self.indx_limit = 99999
        if standard:
            self.kind = in_line[0:6].strip()
            try: self.atom_num = int(in_line[6:11].strip())
            except ValueError:
                if in_line[6:11] == "*****" and prev_at_idx:
                    self.atom_num = prev_at_idx + 1
                else:
                    self.atom_num = in_line[6:11].strip()
            self.atom_name = in_line[12:16].strip()
            self.altloc = in_line[16].strip()
            self.res_name = in_line[17:20].strip()
            if not reference:
                self.chainID = in_line[21].strip()
                self.res_num = int(in_line[22:28].strip())  # 22:26 standard... change also in the writer...
                self.x = float(in_line[30:38].strip())
                self.y = float(in_line[38:46].strip())
                self.z = float(in_line[46:54].strip())
                self.np_coords = array([self.x, self.y, self.z])
                try: self.occupancy = float(in_line[54:60].strip())
                except ValueError: pass
                try: self.tempfactor = float(in_line[60:66].strip())
                except ValueError: pass
                try: self.element = in_line[76:78].strip()
                except ValueError: pass
                try: self.charge = float(in_line[78:80].strip())
                except ValueError: pass
                if prev_at_idx: return self.atom_num
        else:
            in_line = in_line.replace('-', ' -')
            items = in_line.split()
            if len(self.labels) == len(items): add_with_label(items_list=items)
            else:
                self.labels = []
                print(">> Now I'll show the elements of the atom I've found.")
                print(">> For every element of the line identify the correct label from the following list:")
                print("> kind (ATOM/HETATM), atom_num, atom_name, altloc, res_name, chainID, res_num, x, y, z,")
                print("> occupancy, tempfactor, element, charge.")
                print(">> The line is: %s" % in_line)
                for item in items:
                    label = ''
                    while label == '': label = input("> '%s' has label: " % item)
                    self.labels.append(label)
                add_with_label(items_list=items)
            if return_labels: return self.labels
        if zhi:
            # requires a dict of type {zhi_at_name1: res_name1, ...}
            self.res_name_original = self.res_name
            try: self.res_name = zhi[self.atom_name]
            except KeyError:
                print(">> Error: zhi_atom_name %s not found in the input dictionary of residues" % self.atom_name)

    def import_pytraj_atom(self, pytraj_atom, traj="none", frame="none", also_xyz=False):
        from numpy import array
        self.atom_num = pytraj_atom.index + 1
        self.atom_name = pytraj_atom.name
        self.res_name = pytraj_atom.resname
        self.res_num = pytraj_atom.resid + 1
        self.atom_type = pytraj_atom.type
        self.charge = pytraj_atom.charge
        if frame != "none":
            if also_xyz:
                self.x = float(traj.xyz[frame][pytraj_atom.index][0])
                self.y = float(traj.xyz[frame][pytraj_atom.index][1])
                self.z = float(traj.xyz[frame][pytraj_atom.index][2])
            # print("adding coordinates to %s_%d" % (self.atom_name,self.atom_num))
            self.np_coords = array(traj.xyz[frame][pytraj_atom.index])



    def xyz_line_parser(self, in_line, at_num=1, res_num=1, res_name="UNK"):
        items = in_line.split()
        self.kind = "HETATM"
        self.atom_num = at_num
        self.atom_name = items[0]
        self.res_name = res_name
        self.res_num = res_num
        self.x = float(items[1])
        self.y = float(items[2])
        self.z = float(items[3])

    def pdb_line_writer(self, out_file=None, discovery_studio=False):
        self.pdb_line = '{s.kind:<6s}'.format(s=self)
        try:
            if float(self.atom_num) > 99999 or self.atom_num == "*****":
                self.pdb_line += '***** {s.atom_name:>4s}'.format(s=self)
            else:
                if not discovery_studio:
                    self.pdb_line += '{s.atom_num:5d} {s.atom_name:<4s}'.format(s=self)
                else:
                    if len(self.atom_name) == 4 and self.atom_name.startswith("H"):
                        self.tmp_atom_name = self.atom_name[-1] + self.atom_name[:-1]
                        self.atom_name = self.tmp_atom_name
                        self.pdb_line += '{s.atom_num:5d} {s.atom_name:<4s}'.format(s=self)
                    else:
                        self.pdb_line += '{s.atom_num:5d}  {s.atom_name:<3s}'.format(s=self)

        except ValueError:
            self.pdb_line += '***** {s.atom_name:>4s}'.format(s=self)
        if self.altloc: self.pdb_line += '{s.altloc:1s}'.format(s=self)
        else: self.pdb_line += ' '
        self.pdb_line += '{s.res_name:3s} '.format(s=self)
        if self.chainID: self.pdb_line += '{s.chainID:1s}'.format(s=self)
        else: self.pdb_line += ' '
        coords = []
        for var in self.x, self.y, self.z:
            coord = '{:<7.4f}'.format(var).rstrip()
            if len(coord) >= 6 and coord[-1] == '0': coord = coord[:-1]
            if len(coord) >= 7 and "." in coord: coord = '{:<7.3f}'.format(float(coord))
            coords.append(coord)
        if self.res_num <= 9999:
            self.pdb_line += '{s.res_num:4d}    {c[0]:>8s}{c[1]:>8s}{c[2]:>8s}'.format(s=self, c=coords)
        elif self.res_num <= 99999:
            self.pdb_line += '{s.res_num:5d}   {c[0]:>8s}{c[1]:>8s}{c[2]:>8s}'.format(s=self, c=coords)
        elif self.res_num <= 999999:
            self.pdb_line += '{s.res_num:6d}  {c[0]:>8s}{c[1]:>8s}{c[2]:>8s}'.format(s=self, c=coords)
        elif self.res_num <= 9999999:
            self.pdb_line += '{s.res_num:7d} {c[0]:>8s}{c[1]:>8s}{c[2]:>8s}'.format(s=self, c=coords)
        if self.occupancy: self.pdb_line += '{s.occupancy:6.2f}'.format(s=self)
        else: self.pdb_line += '      '
        if self.tempfactor: self.pdb_line += '{s.tempfactor:6.2f}'.format(s=self)
        else: self.pdb_line += '      '
        if self.element: self.pdb_line += '{s.element:2s}'.format(s=self)
        else: self.pdb_line += '  '
        if self.charge: self.pdb_line += '{s.charge:8.5f}'.format(s=self)
        else: self.pdb_line += '        '
        self.pdb_line += '\n'
        if out_file: out_file.write(self.pdb_line)
        else: return self.pdb_line




class Residue(object):
    def __init__(self):
        self.atoms = []
        self.bonds = []
        self.angles = []
        self.dihedrals = []
        self.res_name = ''
        self.res_num = 1

    def add_atom(self, atom_to_add):
        self.atoms.append(atom_to_add)

def parse_pdb_lines_list(pdb_lines_list, np_array=False, skip_res=""):
    atom_list, coords = [], []
    res_list = [Residue()]
    res_idx, line_idx, prev_at_idx = 0, 0, 1
    while line_idx < len(pdb_lines_list):
        line = pdb_lines_list[line_idx]
        if line.startswith('HETATM') or line.startswith('ATOM'):
            atom = Atom()
            prev_at_idx = atom.pdb_line_parser(in_line=line, prev_at_idx=prev_at_idx)
            if skip_res and atom.res_name == skip_res: 
                pass
            elif np_array:
                atom_list.append(atom)
                coords.append([atom.x, atom.y, atom.z])
                res_idx += 1
                res_list.append(Residue())
                res_list[res_idx].add_atom(atom)
                res_list[res_idx].res_name = res_list[res_idx].atoms[0].res_name
                res_list[res_idx].res_num = res_list[res_idx].atoms[0].res_num
            line_idx += 1
        else: 
            line_idx += 1
    if np_array:
        from numpy import array, float64
        return atom_list,res_list, array(coords, dtype=float64)
    else:
        _ = res_list.pop(0)
        return res_list

def convert_pdb_to_xyzr(atoms_list, res_list, coords):
    pdbName="tmp_frame"
    defaultRadii = {}
    specificRadii = {}
    txt=str(ambersiz()).split("\n")
    for i in range(len(txt)):
        line = txt[i]
        # stop reading
        if not line:
            continue	
        # skip empty lines
        if (len(line)<=1):
            continue
        # skip comment line
        if '!' in line:	
            continue
        if (line.startswith('atom__res_radius_')):
            print('>>> Recognized DelPhi header for radii .siz file')
            continue

        listStr = line.split()
        if (len(listStr)==2):
            defaultRadii[listStr[0][0]]=float(listStr[1])
        elif (len(listStr)==3):
            specificRadii['%s_%s'%(listStr[0],listStr[1])]=float(listStr[2])
        else:
            print('>>> Unrecognized record in radii file at line %d, please check file and reload'%i)
            quit()
            
    f = open(('%s.xyzr'%pdbName),'w')
    X = coords

    for i in range(len(atoms_list)):
        resName = res_list[i]
        atomName = atoms_list[i]
        key = '%s_%s'%(atomName.atom_name,resName.res_name)
        if not(key in specificRadii):
            if not(atomName.atom_name in defaultRadii):
                #print('>>> %s %s missing in radii set applying 1.0 Angstrom'%(atomName.atom_name,resName.res_name))
                radius = 1.0
            else:
                radius = defaultRadii[atomName.atom_name]
        else:
            radius = specificRadii[key]
        f.write(('%f %f %f %f\n'%(X[i][0],X[i][1],X[i][2],radius)))
    f.close()
    print('>>> File converted in %s.xyzr'%(pdbName))

def extract_relevant_data(txt):
    txt_regex=txt.replace("\n", "")
    match = re.findall(r"Detected a total of \d* pockets/cavities .+ Pocket detection time",
            txt_regex ,
            flags=re.IGNORECASE)
    if match:
        return list(map(lambda x: x.strip(), match[0].split("<<INFO>>")[1:-1]))

def log_output(string):
    string=string.decode("utf-8")
    #logging.info(string)
    for line in extract_relevant_data(string):
        splitted=line.split()
        logging.info("{:>10} {:>20}".format(int(splitted[1]),float(splitted[3])))

def run_shell_command(command_line, i):
    command_line_args = shlex.split(command_line)
    logging.info(">>>Frame %d"%(i))

    try:
        command_line_process = subprocess.Popen(
            command_line_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        process_output, _ =  command_line_process.communicate()
        #print(process_output)
        log_output(process_output)

    except (OSError, subprocess.CalledProcessError) as error:
        logging.info("Error occured: " + str(error) +"\n")
        logging.info("Subprocess failed at frame %d"%i)
        return False
    else:
        # no exception was raised
        logging.info(b'Subprocess finished\n'.decode("utf-8"))

    return True

def process_log():
    with open(logname, "r") as f:
        txt=f.read()
    end="INFO:root:Subprocess finished"
    txt=txt.replace("\n","").replace(end,"").split("INFO:root:")


    tmp=[]
    for i in list(map(lambda x: x.split(),txt)):
        if i:
            if i[0].startswith(">>>Frame") and int(i[1]) == 0:
                n_pockets=0
                volume=0
            if i[0].startswith(">>>Frame") and int(i[1]) != 0:
                tmp.append([n_pockets,volume])
                n_pockets=0
                volume=0
            else:
                n_pockets+=1
                volume+=float(i[1])

    arr=np.asarray(tmp)
    print("MEAN: {:<15} STD: {:<15}".format(arr[:,1].mean(), arr[:,1].std()))
    plt.hist(arr[:,0], bins=np.arange(np.min(arr[:,0]), np.max(arr[:,0])+1,1), align="left")
    plt.savefig("trench_hist.png")
    plt.close()

traj=pt.iterload(args.traj,args.prmtop)
frames=np.arange(0,len(traj), args.stride)
traj=pt.load(args.traj,args.prmtop, frame_indices=frames)


for i,frame in enumerate(traj):
    cpptraj(args,frames[i])
    with open("tmp_frame.pdb") as in_file:
        atoms_list, res_list, coords = parse_pdb_lines_list(pdb_lines_list=in_file.readlines(), np_array=True)
    convert_pdb_to_xyzr(atoms_list, res_list, coords)
    print(">>> NanoShaper is calculating")
    with open("temp_nano.prm", "w") as f:
        f.write(str(prm_file()))
    run_shell_command("NanoShaper temp_nano.prm", i)
process_log()










