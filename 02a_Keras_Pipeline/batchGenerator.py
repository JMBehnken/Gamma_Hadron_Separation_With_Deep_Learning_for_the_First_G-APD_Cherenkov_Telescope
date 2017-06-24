def batchGenerator(paths_train, label_train, batch_size=500, pg_ratio=0.8):
   
    p_batch_size = int(round(batch_size * (1 - pg_ratio)))
    g_batch_size = int(round(batch_size * pg_ratio))
    
    proton_paths = paths_train[~label_train][:,0]
    gamma_paths = paths_train[label_train][:,0]
    
    proton_batch = batchYielder(proton_paths, p_batch_size)
    gamma_batch = batchYielder(gamma_paths, g_batch_size)
    batch = []
    
    while True:
        event_proton = next(proton_batch)
        event_gamma = next(gamma_batch)
        batch.extend(zip(len(event_proton)*[False], event_proton))
        batch.extend(zip(len(event_gamma)*[True], event_gamma))
        
        if len(batch)>=batch_size:
            random.shuffle(batch)
            batch_500, batch = batch[:batch_size], batch[batch_size:]
            data, label = getMatrix(batch_500)
            X_train = np.array(data)
            y_train = np_utils.to_categorical(label, categories)
        
            yield (X_train/300, y_train)
        
        
def batchYielder(paths, batch_size):

    for path in paths:
        with gzip.open(path) as file:
            while True:
                next_n_lines = list(islice(file, batch_size))
                if not next_n_lines:
                    break
                yield next_n_lines