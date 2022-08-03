


ac_terms = ['Pdb','Group1', 'Group2', 'IntraclashesGroup1', 'IntraclashesGroup2', 'Interaction_Energy', 'Backbone_Hbond', 'Sidechain_Hbond',
'Van_der_Waals', 'Electrostatics', 'Solvation_Polar', 'Solvation_Hydrophobic', 'Van_der_Waals_clashes', 'entropy_sidechain',
'entropy_mainchain', 'sloop_entropy', 'mloop_entropy', 'cis_bond', 'torsional_clash',
 'backbone_clash', 'helix_dipole', 'water_bridge',
 'disulfide', 'electrostatic_kon', 'partial_covalent_bonds', 'energy_Ionisation', 'Entropy_Complex', 'Number_of_Residues',
 'Interface_Residues', 'Interface_Residues_Clashing', 'Interface_Residues_VdW_Clashing', 'Interface_Residues_BB_Clashing']



import pandas as pd
import os
import decimal
import numpy as np

source_dir = 'D:/data2/sars_data_cleaned/moredata/deltaomicron/foldx/'
ac_file_path = source_dir + 'analysecomplex/'
all_files = os.listdir(ac_file_path)

ac_df=pd.DataFrame(columns=ac_terms)

for file in os.listdir(file_path):
    pdb=file[12:-9]
    if file[:11] == 'Interaction':

        with open(file_path+file, 'r') as f:
            result=f.read()

        result = result.splitlines()
        result = result[9].split()

        result[0] = pdb.upper()
        ac_series = pd.Series(dict(zip(ac_terms, result)))
        ac_df = ac_df.append(ac_series, ignore_index=True)

for ind in ac_df.index:
	if 'REPAIR'in 'ind.upper():
		wt_ac=np.array(ac_df.loc[ind]

ac_diff=pd.DataFrame(columns=ac_df.columnds)
for ind in ac_df.index:
	ac_diff.loc[ind] = np.array(ac_df.loc[ind])-wt_ac

ac_diff.to_csv(source_dir + 'analysecomplex_d_result.csv')


st_file_path = source_dir + 'stability/'
files = os.listdir(st_file_path)


st_terms = ['Pdb', 'Total_Energy','Backbone_Hbond', 'Sidechain_Hbond', 'Van_der_Waals',
'Electrostatics', 'Solvation_Polar', 'Solvation_Hydrophobic', 'Van_der_Waals_clashes', 'Entropy_Side_Chain',
'Entropy_Main_Chain', 'Sloop_Entropy', 'Mloop_Entropy', 'Cis_Bond', 'Torsional_Clash', 'Backbone_Clash', 'Helix_Dipole', 'Water_Bridge', 'Disulfide', 'Electrostatic_Kon',
'Partial_Covalent_Bonds', 'Energy_Ionisation', 'Entropy_Complex', 'Residue_Number' ]

st_df=pd.DataFrame(columns=st_terms)

for file in files:
    with open(file_path+file, 'r') as f:
        result=f.read()

    result = result.split()

    result[0] = result[0][2:6].upper() + result[0][6:-4]
    stability_series = pd.Series(dict(zip(st_terms, result)))

    st_df = st_df.append(stability_series, ignore_index=True)

for ind in st_df.index:
	if 'REPAIR' in 'ind.upper():
		wt_st=np.array(ac_df.loc[ind]

st_diff=pd.DataFrame(columns=st_df.columnds)
for ind in st_df.index:
	st_diff.loc[ind] = np.array(st_df.loc[ind])-wt_st

st_diff.to_csv(source_dir + 'stability_d_result.csv')
