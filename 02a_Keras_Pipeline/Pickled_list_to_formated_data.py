import numpy as np
import pickle
import gzip
import os

path = '/fhgfs/users/jbehnken/Conv_Data'
path_new = '/fhgfs/users/jbehnken/np_Conv_Data'

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, 46, 45, 1)).astype(np.float32)
	labels = (np.arange(2) == labels[:,None]).astype(np.float32)
	return dataset, labels

def rewrite(file):
        with gzip.open(path+'/'+file, 'rb') as f:
                data = pickle.load(f)
                pic, lab = zip(*data)
                pic, lab = reformat(np.array(pic), np.array(lab))

        data_dict={'Image':pic, 'Label':lab}

        with gzip.open(path_new+'/'+file, 'wb') as f:
                pickle.dump(data_dict, f)

from multiprocessing import Pool
p = Pool()
p.map(rewrite, os.listdir(path))
