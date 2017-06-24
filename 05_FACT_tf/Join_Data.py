import h5py
import numpy as np
import pandas as pd

keys = ['event_num', 'night', 'run_id', 'theta', 'theta_deg', 'theta_deg_off_1', 'theta_deg_off_2', 'theta_deg_off_3', 'theta_deg_off_4', 'theta_deg_off_5', 'theta_off_1', 'theta_off_1_rec_pos', 'theta_off_2', 'theta_off_2_rec_pos', 'theta_off_3', 'theta_off_3_rec_pos', 'theta_off_4', 'theta_off_4_rec_pos', 'theta_off_5', 'theta_off_5_rec_pos', 'theta_rec_pos']

with h5py.File('/net/big-tank/POOL/projects/fact/datasets/Crab1314_darknight_std_analysis_0.17.2.hdf5', 'r') as f:
    data  = []
    for key in keys:
        data.append(np.array(f['events'][key]))
        
data_2 = list(map(list, zip(*data)))

df = pd.DataFrame(data_2)
df.columns = keys

df.to_csv('FACT_Data_to_merge.csv', chunksize=1000, index=False)

df_my = pd.read_csv('Data_to_merge.csv')
df_fact = pd.read_csv('FACT_Data_to_merge.csv')

df_my.columns = ['night', 'run_id', 'event_num', 'proton_prediction', 'gamma_prediction']

df_merged = pd.merge(df_my, df_fact, how='inner', on=['event_num', 'night', 'run_id'])

df_merged.to_csv('Merged_Data.csv', chunksize=1000, index=False)