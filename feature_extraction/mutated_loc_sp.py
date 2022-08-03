



import operator

#To produce interface distance extracting from total residue distance and interface residues

aa_dict={ 'ARG':'R', 'HIS':'H', 'LYS':'K', 'ASP':'D', 'GLU': 'E', 'SER':'S', 'THR': 'T', 'ASN': 'N', 'GLN':'Q',
          'CYS':'C', 'GLY':'G', 'PRO':'P', 'ALA':'A', 'VAL':'V','ILE':'I','LEU':'L', 'MET':'M', 'PHE':'F', 'TYR':'Y', 'TRP':'W'}

aa_dict=dict(zip(aa_dict.values(), aa_dict.keys()))

import os
import pandas as pd

aas_df=pd.read_csv('D:/data2/SARS2-SPIKE/interface_residue_descriptors/interface_residues_sars2.csv')
potential_dir = 'D:/data2/sars_data_cleaned/cnn_data/statistical_potential/'
dist_dir= 'D:/data2/sars_data_cleaned/cnn_data/structure_distance/'
output_dir = 'D:/data2/SARS2-SPIKE/interface_residue_descriptors/mutated_location_statistical_potential_d/'

wt_potential_df = pd.read_csv(potential_dir + '6M0J_Repair_potential.csv')
wt_potential_df.set_index('Unnamed: 0', inplace=True)

wt_if_aas=aas_df.loc[0, 'interface_residues'].replace(" ", "").split(",")

for a in range(len(wt_if_aas)):
    aa=wt_if_aas[a]
    aa=aa[1] + "_" + aa_dict[aa[0]]+aa[2:]
    wt_if_aas[a] = aa

wt_if_df=pd.DataFrame(index=wt_if_aas, columns=wt_if_aas)
for ind in wt_if_df.index:
    for col in wt_if_df.columns:
        wt_if_df.loc[ind, col] = wt_potential_df.loc[ind, col]


wt_resnames=[]
for aa in wt_if_aas:
    resname=aa[:1] + aa[5:]
    wt_resnames.append(resname)

wt_if_df.index=wt_resnames
wt_if_df.columns=wt_resnames

if_resnames=wt_resnames

missed=[]

wt_idx=[]
for ind in wt_potential_df.index:
    wt_idx.append(ind[:1] + ind[5:])

wt_potential_df.index=wt_idx
wt_potential_df.columns=wt_idx

for ind in aas_df.index[1:]:
    filename = aas_df.loc[ind, 'PDB_code']
    if '{}_mutloc_distance.csv'.format(filename) not in os.listdir(output_dir):

		dist_df = pd.read_csv(dist_dir +filename + '_distance.csv')
		dist_df.set_index('Unnamed: 0', inplace=True)
		potential_df = pd.read_csv(potential_dir + filename + '_potential.csv')
		potential_df.set_index('Unnamed: 0', inplace=True)
		df_resnames=[]
		for col in potential_df.columns:
			df_resnames.append(col[:1] + col[5:])

		potential_df.columns=df_resnames
		potential_df.index=df_resnames
		dist_df.index=df_resnames
		dist_df.columns=df_resnames


		#mutated location residues

		mutloc = filename.split("_")[1] + filename.split("_")[2][1:-1]
		mutloc_resnames=[]

		mutloc_resi= []
		mutloc_resi_dict={}

		for resi in enumerate(list(dist_df.loc[mutloc])):
			mutloc_resi_dict[dist_df.columns[resi[0]]] = resi[1]

		mutloc_resi_sorted=sorted(mutloc_resi_dict.items(), key=operator.itemgetter(1))


		for resi in mutloc_resi_sorted[:21]:
			mutloc_resi.append(resi[0])

		mutloc_resnames.extend(mutloc_resi)

		st_df = pd.DataFrame(columns=mutloc_resnames, index=mutloc_resnames)


		for resname in mutloc_resnames:
			for resname2 in mutloc_resnames:
				st_df.loc[resname, resname2] = float(potential_df.loc[resname, resname2]) - float(wt_potential_df.loc[resname, resname2])


		st_df.to_csv(output_dir + filename + '_mutloc_d_potential.csv')


			


