from multiprocessing import Pool
import numpy as np
import operator
import random
import pickle
import gzip
import json
import os

# Important variables
mc_data_path = '/net/big-tank/POOL/projects/fact/simulation/photon_stream/fact_tools/v.0.18.0/'
id_position_path = '/home/jbehnken/06_FACT_Pipeline/01_hexagonal_position_dict.p'
temporary_path = '/fhgfs/users/jbehnken/01_Data/99_Temporary'
processed_data_path = '/fhgfs/users/jbehnken/01_Data/01b_MC_Data_flat'

def getMetadata():
    '''
    Gathers the file paths of the training data
    '''
    # Iterate over every file in the subdirs and check if it has the right file extension
    file_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(os.path.expanduser(mc_data_path)) for file in fileName if '.json' in file]
    return file_paths


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, 46, 45, 1)).astype(np.float32)
    labels = (np.arange(2) == labels[:,None]).astype(np.float32)
    return dataset, labels

file_paths = getMetadata()
id_position = pickle.load(open(id_position_path, "rb"))

data = []
num = 0
for path in file_paths:
    with gzip.open(path) as file:
        # Gamma=True, Proton=False
        label = True if 'gamma' in path else False
        
        for line in file:
            try:
                event_photons = json.loads(line.decode('utf-8'))['PhotonArrivals_500ps']
            
                event = []
                input_matrix = np.zeros([46,45,100])
                for i in range(1440):
                    event.extend(event_photons[i])
                    
                    x, y = id_position[i]
                    for value in event_photons[i]:
                        input_matrix[int(x)][int(y)][value-30] += 1
                        
                count_dict = {}
                for i in event:
                    if i in count_dict.keys():
                        count_dict[i] += 1
                    else:
                        count_dict[i] = 1
                        
                sorted_dict = sorted(count_dict.items(), key=operator.itemgetter(1), reverse=True)
                one = sorted_dict[0][0]
                two = sorted_dict[1][0]
                        
                if one==two or one==two+1 or one==two-1 and one>=20 and one<=60:
                    lim_min = one-32 if one>32 else 0
                    lim_max = one-28
                
                    input_matrix = np.sum(input_matrix[:,:,lim_min:lim_max], axis=2)
                    data.append([input_matrix, label])
            
            
            
                if len(data)%1000 == 0 and len(data)!=0:
                    pic, lab = zip(*data)
                    pic, lab = reformat(np.array(pic), np.array(lab))
                    data_dict={'Image':pic, 'Label':lab}
                    
                    with gzip.open( temporary_path + "/PhotonArrivals_500ps_"+str(num)+".p", "wb" ) as data_file:
                        pickle.dump(data_dict, data_file)
                    data = []
                    num += 1
               
            except:
                pass
                
# Load pickled data and split it into pictures and labels
def load_data(file):
    with gzip.open(temporary_path+'/'+file, 'rb') as f:
        data_dict = pickle.load(f)
    pic = data_dict['Image']
    lab = data_dict['Label']
    return (pic, lab)

# Pool-load pickled data and split it into pictures and labels (list)
p = Pool()
data = p.map(load_data, os.listdir(temporary_path))
pics, labs = zip(*data)
del data, p

# Concatenate the data to a single np.array
pic = np.concatenate(pics)
lab = np.concatenate(labs)
del pics, labs


# Values to standardize the data
mean = np.mean(pic)
std = np.std(pic)
print(mean, std)


# Randomize and split the data into train/validation/test dataset
p = np.random.permutation(len(pic))
all_pics = pic[p]
all_labels = lab[p]
del p, pic, lab

def save_data(i):
    pics_batch = all_pics[(i-1)*1000:i*1000]
    labels_batch = all_labels[(i-1)*1000:i*1000]
    
    data_dict={'Image':(pics_batch-mean)/std, 'Label':labels_batch}
    with gzip.open(processed_data_path + '/PhotonArrivals_500ps_{}.p'.format(i), 'wb') as f:
        pickle.dump(data_dict, f)
        
num_files = len(os.listdir(temporary_path))
p = Pool()
data = p.map(save_data, range(1,num_files+1))


