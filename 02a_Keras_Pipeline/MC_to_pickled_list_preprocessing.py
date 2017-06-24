exec(open('getMetadata.py').read())

import pickle
import gzip

df = getMetadata(load_metadata=True)
id_position = pickle.load(open("position_dict.p", "rb"))

data = []
num = 0
for elem in df.values:
    with gzip.open(elem[0]) as file:
        for line in file:
            event_photons = json.loads(line.decode('utf-8'))['PhotonArrivals_500ps']
            
            input_matrix = np.zeros([46,45])
            for i in range(1440):
                x, y = id_position[i]
                input_matrix[int(x)][int(y)] = len(event_photons[i])
            
            data.append([input_matrix, elem[2]])
            
            if len(data)%10000 == 0:
                with gzip.open( "/fhgfs/users/jbehnken/Conv_Data/PhotonArrivals_500ps_"+str(num)+".p", "wb" ) as data_file:
                    pickle.dump(data, data_file)
                data = []
                num += 1