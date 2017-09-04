import pandas as pd
import os

p='/fhgfs/users/jbehnken/make_Data/architectures'
folders=[os.path.join(p, path) for path in os.listdir(p)]

for folder in folders:
	df=pd.read_csv(folder+'/'+folder.split('_')[-1]+'_Hyperparameter.csv')
	print(df['Auc'].max())
