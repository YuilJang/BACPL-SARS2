
import pandas as pd
import os



a =pd.read_csv("6M0J_A_PSSM.csv")
a.set_index('pdb_num', inplace=True)


e = pd.read_csv("6M0J_E_PSSM.csv")
e.set_index('pdb_num', inplace=True)


source_dir = os.getcwd() + 'buildmodel/'
pssm_d=pd.DataFrame(columns= ['score_diff'])
for file in os.listdir(source_dir):

    if file[:4] == '6m0j' and 'Repair' not in file:

        #single mutation
        if len(file[:-4].split('_')) == 2:

            mut=file[:-4].split('_')[1]
            chain=mut[1]
            ori_aa=mut[0]
            mut_aa=mut[-1]
            seq=int(mut[2:-1])

            if chain == 'A':
                ori_score=a.loc[seq, ori_aa]
                mut_score=a.loc[seq, mut_aa]
                score_d= mut_score- ori_score
            elif chain == 'E':

                score_d = e.loc[seq, ori_aa] - e.loc[seq, mut_aa]

            pssm_d.loc[file[:-4]] = score_d
        #multiplt mutation
        if len(file[:-4].split('_')) > 2:
            muts=file[:-4].split('_')[1:]

            score_ds=[]

            for mut in muts:
                chain = mut[1]
                ori_aa = mut[0]
                mut_aa = mut[-1]
                seq=int(mut[2:-1])

                if chain == 'A':
                    score_d = a.loc[seq, ori_aa] - a.loc[seq, mut_aa]
                    score_ds.append(score_d)
                elif chain == 'E':
                    score_d = e.loc[seq, ori_aa] - e.loc[seq, mut_aa]
                    score_ds.append(score_d)

            average_score_d = sum(score_ds)/len(score_ds)
            pssm_d.loc[file[:-4]] = average_score_d




pssm_d.to_csv(source_dir + 'pssm_score_d.csv')