import os
import argparse
import sys

#Simple script to convert RED Server mol2 charges to the Amber format

def read(path):
	with open(path, "r") as file:
		atoms_section = False
		charge_dic = {}
		for i, line in enumerate(file.readlines()):
			splitted = line.split()
			if i == 6:
				#Number of atoms for a basic check
				n = int(splitted[0])
			if splitted[0] == '@<TRIPOS>BOND':
				atoms_section = False
			if atoms_section:
				#Create dictionary of atoms and charges
				charge_dic[splitted[1]] = str(splitted[8])
			if splitted[0] == '@<TRIPOS>ATOM':
				atoms_section = True
		assert n == len(charge_dic), f"Number of atoms {n} and charges {len(charge_dic)} caused a problem. Resolve manually"
	return charge_dic


def write(path, out, charges):
	filename=os.path.basename(path).rsplit('.', 1)[0]
	with open(path, "r") as ambermol:
		with open(os.path.join(out, filename+".charged.mol2"), "w") as newmol:
			atoms_section = False
			for line in ambermol.readlines():
				splitted = line.split()
				try:
					if splitted[0] == '@<TRIPOS>BOND':
						atoms_section = False
					if atoms_section:
						#Compose line with correct charges
						correct_charge = charges[splitted[1]]
						tmp_line = line[0:-9]
						newmol.write(tmp_line+correct_charge+"\n")
					if splitted[0] == '@<TRIPOS>ATOM':
						newmol.write(line)
						atoms_section = True
					if not atoms_section:
						newmol.write(line)
				except IndexError:
					newmol.write("\n")


def main(args):
	charges = read(args.inp_red)
	write(args.inp_amber, args.out, charges)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description='Automatic charges filling from RED mol2 to Antechamber mol2.')
	parser.add_argument("-ir", "--inputred", type=str, dest='inp_red',
						help="Input path to RED mol2.")
	parser.add_argument("-ia", "--inputamber", type=str, dest='inp_amber',
						help="Input path to antechamber mol2.")
	parser.add_argument("-o", "--output", type=str, dest='out', default=(os.path.abspath(os.path.dirname(sys.argv[0]))),
						help="Output path.")

	args = parser.parse_args()

	try:
		main(args)
	except OSError as error:
		raise error
