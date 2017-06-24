import pickle
import gzip
import numpy as np
import os
import random
from multiprocessing import Pool

# Path to preprocessed data
path = '/fhgfs/users/jbehnken/np_Conv_Data'


# Load pickled data and split it into pictures and labels
def load_data(file):
    with gzip.open(path+'/'+file, 'rb') as f:
        data_dict = pickle.load(f)
    pic = data_dict['Image']
    lab = data_dict['Label']
    return (pic, lab)

# Pool-load pickled data and split it into pictures and labels (list)
p = Pool()
data = p.map(load_data, os.listdir(path))
pics, labs = zip(*data)
del data, p

# Concatenate the data to a single np.array
pic = np.concatenate(pics)
lab = np.concatenate(labs)
del pics, labs


# Randomize and split the data into train/validation/test dataset
p = np.random.permutation(len(pic))
all_pics = pic[p]
all_labels = lab[p]
del p, pic, lab

def save_data(i):
    pics_batch = all_pics[(i-1)*10000:i*10000]
    labels_batch = all_labels[(i-1)*10000:i*10000]
    
    data_dict={'Image':pics_batch, 'Label':labels_batch}
    with gzip.open('/fhgfs/users/jbehnken/rand_Conv_Data/PhotonArrivals_500ps_{}.p'.format(i), 'wb') as f:
        pickle.dump(data_dict, f)
        
p = Pool()
data = p.map(save_data, range(1,243))