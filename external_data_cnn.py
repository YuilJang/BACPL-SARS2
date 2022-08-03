

#모델 로드
torch.manual_seed(42)
cnn_model = CNN()
cnn_model = torch.load('C:/Users/JangYuil/Desktop/distance_matrix_cnn/trainset_cnn_model.pt')



mulmut_dir = 'D:/data2/sars_data_cleaned/moredata/including_mulmut/minimize/structure_distance/mut-wt/'
mulmut_data_paths=[]
mulmut_label_df = pd.read_csv('D:/data2/sars_data_cleaned/moredata/including_mulmut/minimize/including_mulmut2.csv')
mulmut_label_df.set_index('PDB_code', inplace=True)

#custum dataset

import torchvision.transforms as transforms
data_transformer = transforms.Compose([transforms.ToTensor()])

mulmut_paths=[]
for mtx in os.listdir(mulmut_dir):
    if mtx[:4] == '6m0j':

        if mtx[:-23].upper() in list(mulmut_label_df.index):
            mulmut_path=mulmut_dir + mtx
            mulmut_paths.append(mulmut_path)


print(" size: {}".format(len(mulmut_paths)))

#Define dataset

class mulmut_CustomDataset(Dataset):
    def __init__(self, data_paths,transform=False):
        self.data_paths = data_paths

        self.transform = transform
        
        
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_filepath = self.data_paths[idx]
        mtx=pd.read_csv(data_filepath)
        mtx.set_index('Unnamed: 0', inplace=True)
        mtx=np.array(mtx)
        mtx = torch.FloatTensor(mtx)
       
        
        label = float(mulmut_label_df.loc[data_filepath.split('/')[-1][:-23].upper(), 'ddG'])
        label= torch.tensor(label, dtype=torch.float)
        
                
        return mtx, label
    
#######################################################
#                  Create Dataset
#######################################################

mulmut_dataset =mulmut_CustomDataset(mulmut_paths, data_transformer)


#######################################################
#                  Define Dataloaders
#######################################################

mulmut_loader = DataLoader(
    mulmut_dataset, shuffle=True
)


# In[29]:


torch.manual_seed(42)
mulmut_loss, mulmut_accuracy, mulmut_pred = evaluate(cnn_model, mulmut_loader)


# In[30]:


#예측값 저장

mulmut_inds=[]

for mulmut_path in mulmut_paths:
	ind=mulmut_path.split('/')[-1][:-23].upper()
	mulmut_inds.append(ind)
	
ddg_preds=[]

for pred in mulmut_pred:
	pred=float(pred.to('cpu'))
	ddg_preds.append(pred)
	

mulmut_cnn_pred=pd.DataFrame(columns=['ddG_expr', 'ddG_pred'])

for i in range(len(mulmut_inds)):
	mulmut_cnn_pred.loc[mulmut_inds[i], 'ddG_pred'] = ddg_preds[i]
	mulmut_cnn_pred.loc[mulmut_inds[i], 'ddG_expr'] = mulmut_label_df.loc[mulmut_inds[i], "ddG"]
	
# mulmut_cnn_pred.to_csv('C:/Users/JangYuil/Desktop/distance_matrix_cnn/mulmut_cnn_pred.csv')