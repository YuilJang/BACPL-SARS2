
import os

source_dir=os.getcwd() + "/"

pdb_dir=source_dir + 'pdb/'

temperature = 310.18

os.chdir(pdb_dir)

if 'stability' not in os.listdir(source_dir):
	os.system('mkdir {}stability'.format(source_dir))

if 'analysecomplex' not in os.listdir(source_dir):
	os.system('mkdir {}analysecomplex'.format(source_dir))

for file in os.listdir(pdb_dir):
	
	if file[-3:] == 'pdb':

		os.system('foldx_20221231 --command=Stability --pdb={} --output-dir={}stability/ --temperature={}'.format(file, source_dir,temperature))
		os.system('foldx_20221231 --command=AnalyseComplex --pdb={} --analyseComplexChains=E --output-dir={}analysecomplex/ --temperature={}'.format(file, source_dir,temperature))
