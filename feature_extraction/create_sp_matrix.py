
#Create statistical potential matrix 
aa_dict={'ARG': 'R', 'HIS': 'H', 'LYS': 'K', 'ASP':'D', 'GLU': 'E', 'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q',
         'CYS':'C', 'SEC': 'U', 'GLY': 'G', 'PRO': 'P', 'ALA': 'A', 'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'PHE': 'F',
         'TRP': 'W', 'TYR': 'Y', 'VAL':'V'}

import pandas as pd
import os


idx_dir = 'C:/Users/JangYuil/Desktop/descriptor/aaindex/aa_index2/'

potential_E_dir = idx_dir
dist_0_5 = pd.read_csv(potential_E_dir+'Distance-dependent statistical potential (contacts within 0-5 Angstrooms).csv')
dist_5_7 = pd.read_csv(potential_E_dir+ 'Distance-dependent statistical potential (contacts within 5-7.5 Angstrooms).csv')
dist_7_10 =  pd.read_csv(potential_E_dir+ 'Distance-dependent statistical potential (contacts within 7.5-10 Angstrooms).csv')
dist_10_12 =  pd.read_csv(potential_E_dir+ 'Distance-dependent statistical potential (contacts within 10-12 Angstrooms).csv')
dist_12 =  pd.read_csv(potential_E_dir+ 'Distance-dependent statistical potential (contacts longer than 12 Angstrooms).csv')

dfs=[dist_0_5, dist_5_7, dist_7_10, dist_10_12, dist_12]

for d in dfs:

    d.set_index('AA', inplace=True)



source_dir= os.getcwd()
distance_matrix_dir= source_dir + 'structure_distance/'
output_dir=source_dir + 'statistical_potential/'


if 'statistical_potential' not in os.listdir(source_dir):
    os.mkdir(output_dir)


dist_matrices=[]
for file in os.listdir(distance_matrix_dir):
    if file[:-13] + '_potential.csv' not in os.listdir(output_dir):
        dist_matrices.append(file)

for mtrx in dist_matrices:

    mtrx_df=pd.read_csv(distance_matrix_dir+mtrx)
    mtrx_df.set_index('Unnamed: 0', inplace=True)
    seq = mtrx_df.index
    potE_df=pd.DataFrame()
    for resi in seq:
        for resi2 in seq:
            dist = mtrx_df.loc[resi, resi2]
            aa = aa_dict[resi[2:5]]
            aa2 = aa_dict[resi2[2:5]]

            if 0 <= dist and dist < 5:
                potE_df.loc[resi, resi2] = dist_0_5.loc[aa,aa2]
            elif 5<= dist and dist < 7.5 :
                potE_df.loc[resi, resi2] = dist_5_7.loc[aa,aa2]
            elif 7.5 <= dist and dist < 10:
                potE_df.loc[resi, resi2] = dist_7_10.loc[aa,aa2]
            elif 10 <= dist and dist <12:
                potE_df.loc[resi,resi2] = dist_7_10.loc[aa,aa2]
            elif 12<= dist and dist <15:
                potE_df.loc[resi, resi2] = dist_12.loc[aa,aa2]
            elif 15<= dist:
                potE_df.loc[resi, resi2] = 0

    potE_df.to_csv(output_dir + mtrx[:-13] + '_potential.csv')
