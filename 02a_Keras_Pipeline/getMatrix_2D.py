def getMatrix(data_batch):
    
    import pickle
    id_position = pickle.load( open( "position_dict.p", "rb" ))
    
    
    data = []
    label = []    
    
    for event in data_batch:
        event_photons = json.loads(event[1].decode('utf-8'))['PhotonArrivals_500ps']

        input_matrix = np.zeros([91,45])
        for i in range(1440):
            x, y = id_position[i]
            input_matrix[x][y] = len(event_photons[i]) 
        data.append(input_matrix)

        label.append(event[0])
    return data, label