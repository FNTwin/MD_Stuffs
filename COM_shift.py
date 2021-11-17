#!/usr/bin/env python3
from MyUtil.ChemObj import parse_pdb_lines_list, Atom
import sys
import math
import os.path
from argparse import Action, ArgumentParser
import numpy as np

class ArgparseLow(Action):
    def __call__(self, parser, args, values, option_string=None):
        setattr(args, self.dest, values.lower())

parser = ArgumentParser(description="""Script to shift the positions of a membrane and a 
    nanoparticle (or another system) on one axis. By default the membrane COM will
    be shifted to origin and the nanoparticle will be shifted along the selected axis.
    This can be changed with the --switch flag.
    """)
groupA = parser.add_argument_group('INPUT OPTIONS', '')
groupA.add_argument("-i", 
            type=str, 
            dest="membrane",
            help="Membrane PDB file")
groupA.add_argument("-d", 
            type=str, 
            dest="drug",
            help="Drug to position PDB file")
groupA.add_argument("-shift", 
            type=float, 
            dest="shift",
            help="Shift position along the selected-axis to the chosen coordinate")
groupB = parser.add_argument_group('OTHER OPTIONS', '')
groupB.add_argument("--ax", 
            type=str, 
            dest="ax",
            default="z",
            metavar="Z",
            action=ArgparseLow,
            choices=["x", "y", "z", "X", "Y", "Z"],
            help="Select the axis to shift the system")
groupB.add_argument("--membrane-atm",
            type=str,
            dest="mem_atm",
            metavar="N31",
            default="N31",
            nargs="+",
            help="Membrane atoms to calculate center of geometry [Def. N31]")
groupB.add_argument("--membrane-all",
            default=False,
            dest="mem_all",
            action="store_true",
            help="Flag to calculate COG of membrane with all the atoms")
groupB.add_argument("--drug-atm",
            type=str,
            dest="drug_atm",
            metavar="Au",
            default="Au",
            nargs="+",
            help="Nanoparticle or drug atoms to calculate center of geometry [Def. Au]")
groupB.add_argument("--drug-all",
            default=False,
            dest="drug_all",
            action="store_true",
            help="Flag to calculate COG of NP/drug with all the atoms")
groupB.add_argument("--switch",
            default=False,
            dest="switch",
            action="store_true",
            help="Flag to switch membrane and NP position")


opt = parser.parse_args()

def move_selection(atom_list, shift_x, shift_y, shift_z):
    for atom in atom_list:
        atom.traslate(shift_x, shift_y, shift_z)

def coord_selection(atom_list):
    return np.array([[atom.x,atom.y,atom.z] for atom in atom_list])

def create_correct_ax_array(shift):
    if opt.ax == "x":
        index=0
    if opt.ax == "y":
        index=1
    if opt.ax == "z":
        index=2
    tmp=np.zeros(3)
    tmp[index]=shift
    return tmp


with open(opt.membrane) as in_file:
    mem_atoms_list,mem_coord=parse_pdb_lines_list(pdb_lines_list=in_file.readlines(), np_array=True,
                                              skip_res="WAT,Na+,Cl-")
    selection_mem=range(0,len(mem_atoms_list))

with open(opt.drug) as in_file:
    drug_atoms_list,drug_coord=parse_pdb_lines_list(pdb_lines_list=in_file.readlines(), np_array=True,
                                              skip_res="WAT,Na+,Cl-")
    selection_drug=range(0,len(drug_atoms_list))


if not opt.mem_all:
    selection=[index for index in range(0,len(mem_atoms_list)) if mem_atoms_list[index].atom_name in opt.mem_atm]
com_shift=mem_coord[selection_mem].mean(axis=0)

if not opt.drug_all:
    selection_drug=[index for index in range(0,len(drug_atoms_list)) if drug_atoms_list[index].atom_name in opt.drug_atm]
d_com=drug_coord[selection_drug].mean(axis=0)

if not opt.switch:
    move_selection(mem_atoms_list, *-com_shift)
    move_selection(drug_atoms_list, *(-d_com+create_correct_ax_array(opt.shift)))
else:
    move_selection(mem_atoms_list, *-com_shift+create_correct_ax_array(opt.shift))
    move_selection(drug_atoms_list, *(-d_com))


with open(opt.membrane.rsplit(".")[0]+"_shifted.pdb", "w") as out_file:
    for atom in mem_atoms_list:
        atom.pdb_line_writer(out_file=out_file)

with open(opt.drug.rsplit(".")[0]+"_shifted.pdb", "w") as out_file:
    for atom in drug_atoms_list:
        atom.pdb_line_writer(out_file=out_file)


