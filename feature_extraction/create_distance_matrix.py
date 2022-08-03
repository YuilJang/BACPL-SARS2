

#Calculate distance matrix at interface residue

aa_dict={ 'ARG':'R', 'HIS':'H', 'LYS':'K', 'ASP':'D', 'GLU': 'E', 'SER':'S', 'THR': 'T', 'ASN': 'N', 'GLN':'Q',
          'CYS':'C', 'GLY':'G', 'PRO':'P', 'ALA':'A', 'VAL':'V','ILE':'I','LEU':'L', 'MET':'M', 'PHE':'F', 'TYR':'Y', 'TRP':'W'}

aa_dict = {v:k for k,v in aa_dict.items()}


from Bio.PDB.PDBParser import PDBParser
import pandas as pd
import os

p = PDBParser(PERMISSIVE=1)


source_dir= 'D:/data2/SARS2-SPIKE/'
pdb_dir = source_dir+'buildmodel/'
output_dir = source_dir + 'structure_distance/'

if_residues = ['SA19', 'IA21', 'EA23', 'QA24', 'TA27', 'FA28', 'DA30', 'KA31', 'HA34', 'EA35', 'EA37', 'DA38', 'YA41', 'QA42', 'LA45', 'QA76', 'LA79', 'MA82', 'YA83', 'QA325', 'GA326', 'NA330', 'KA353', 'GA354', 'DA355', 'RA357', 'AA386', 'RA393', 'RE403', 'EE406', 'KE417', 'GE446', 'GE447', 'YE449', 'YE453', 'LE455', 'FE456', 'YE473', 'AE475', 'GE476', 'SE477', 'EE484', 'GE485', 'FE486', 'NE487', 'CE488', 'YE489', 'FE490', 'QE493', 'SE494', 'GE496', 'QE498', 'TE500', 'NE501', 'GE502', 'VE503', 'GE504', 'YE505', 'QE506']

if_new_idxs = []

for resi in if_residues:
    idx=resi[1] + "_" + aa_dict[resi[0]] + resi[2:]
    if_new_idxs.append(idx)


if 'structure_distance' not in os.listdir(source_dir):
    os.mkdir(output_dir)

pdb_list = []

for file in os.listdir(pdb_dir):
    if file[:2] != 'WT' and file[-3:] == 'pdb':

        pdb_list.append(file)
    else: pass


for pdb in pdb_list:


    filename= pdb[:-4] + "_distance.csv"
    if filename not in os.listdir(output_dir):

        structure = p.get_structure('structure', pdb_dir + pdb)

        model = structure[0]

        dist_matrix=pd.DataFrame()

        for new_idx in if_new_idxs:
            for new_idx2 in if_new_idxs:
                resi1=model[new_idx[0]][(" ", int(new_idx[5:]), " ")]["CA"]
                resi2=model[new_idx2[0]][(" ", int(new_idx2[5:]), " ")]["CA"]

                dist_matrix.loc[new_idx, new_idx2] = resi1 - resi2


        dist_matrix.to_csv(output_dir + filename)


