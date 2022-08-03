




import freesasa
import os
import pandas as pd
import decimal

dirr= os.getcwd()
pdb_path = dirr + 'pdb/'
wt_pdb_path= pdb_path
pdb_list = os.listdir(pdb_path)
wt_pdbs=[]


keys = []

for a in pdb_list:
       if a[-3:] == 'pdb' and 'WT' not in a:
              keys.append(a)

for b in os.listdir(wt_pdb_path):
       if 'Repair' in b and b[-3:] == 'pdb':
              wt_pdbs.append(b)

sasa_dict = {}
polar_dict = {}
apolar_dict = {}
sasa_diff_dict = {}

for wt in wt_pdbs:
    wt_struc = freesasa.Structure(wt_pdb_path + wt)
    wt_result = freesasa.calc(wt_struc)

    wt_totalsasa = wt_result.totalArea()
    sasa_dict[wt.upper()[:4]] = wt_totalsasa

    area_classes = freesasa.classifyResults(wt_result, wt_struc)
    polar_dict[wt.upper()[:4]] = area_classes['Polar']
    apolar_dict[wt.upper()[:4]] = area_classes['Apolar']
    sasa_diff_dict[wt.upper()[:4]] = 0

    pdb_per_wt=[]
        
    
    for pdb in keys:
        if pdb[:4].upper() == '6M0J':
    
                     pdb_per_wt.append(pdb)

    for mut in pdb_per_wt:

              structure = freesasa.Structure(pdb_path+mut)
              result = freesasa.calc(structure)

              totalsasa = result.totalArea()
              sasa_dict[mut.upper()[:-4]] = totalsasa

              area_classes = freesasa.classifyResults(result, structure)
              polar_dict[mut.upper()[:-4]] = area_classes['Polar']
              apolar_dict[mut.upper()[:-4]] = area_classes['Apolar']
              sasa_diff_dict[mut.upper()[:-4]] = decimal.Decimal(totalsasa) - decimal.Decimal(wt_totalsasa)



sasa_df = pd.DataFrame().from_dict(sasa_dict, orient='index')
polar_seri = pd.Series(polar_dict)
apolar_seri = pd.Series(apolar_dict)
sasa_diff_seri = pd.Series(sasa_diff_dict)

sasa_df['Polar_sasa'] = polar_seri
sasa_df['Apolar_sasa'] = apolar_seri
sasa_df['sasa_diff'] = sasa_diff_seri


sasa_df.to_csv(dirr + 'freesasa_result.csv')