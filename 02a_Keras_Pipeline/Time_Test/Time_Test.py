import pickle
import gzip
import numpy as np
import os
from multiprocessing import Pool


from time import time


path = '/fhgfs/users/jbehnken/np_Conv_Data'


def load_data(file):
	with gzip.open(path+'/'+file, 'rb') as f:
		data_dict = pickle.load(f)
	pic = data_dict['Image']
	lab = data_dict['Label']
	return (pic, lab)
    
    
files = 250 # max 242

p_start = time()
p = Pool()
data = p.map(load_data, os.listdir(path)[:files])
pics, labs = zip(*data)
del data, p
p_end = time()
p_dur = p_end - p_start


c_start = time()
pic = np.concatenate(pics)
lab = np.concatenate(labs)
del pics, labs
c_end = time()
c_dur = c_end - c_start


r_start = time()
p = np.random.permutation(len(pic))
valid_dataset = pic[p][:10000]
valid_labels = lab[p][:10000]
test_dataset = pic[p][10000:60000]
test_labels = lab[p][10000:60000]
train_dataset = pic[p][60000:]
train_labels = lab[p][60000:]
del p, pic, lab
r_end = time()
r_dur = r_end - r_start
a_dur = r_end - p_start



import csv
with open('Time_Test.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow([p_dur, c_dur, r_dur, a_dur])