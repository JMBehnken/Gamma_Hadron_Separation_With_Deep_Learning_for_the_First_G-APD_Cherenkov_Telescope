def getMatrix(data_batch):
    data = []
    label = []


    for event in data_batch:
        event_photons = json.loads(event[1].decode('utf-8'))['PhotonArrivals_500ps']

        input_matrix = np.zeros([1440,100])
        for i in range(1440):
            for j in event_photons[i]:
                input_matrix[i][j-30] += 1
        data.append(input_matrix.flatten())

        label.append(event[0])
    return data, label