def getMetadata(load_metadata=True):

    file_path = 'File_event_count.csv'
    
    if load_metadata:
        df = pd.read_csv(file_path)
        
    else:
        # Path to the directory containing subdirectories and all datafile
        main_path = '/net/big-tank/POOL/projects/fact/simulation/photon_stream/fact_tools/v.0.18.0/'

        # Iterate over every file in the subdirs and check if it has the right file extension
        file_paths = [os.path.join(dirPath, file) for dirPath, dirName, fileName in os.walk(os.path.expanduser(main_path)) for file in fileName if '.json' in file]
        
        # Count numbers of files in every subdir
        proton_files = []
        gustav_files = []
        werner_files = []
        fehler_files = []

        for file in file_paths:
            if 'proton' in file:
                proton_files.append(file)
            elif 'gustav' in file:
                gustav_files.append(file)
            elif 'werner' in file:
                werner_files.append(file)
            else: fehler_files.append(file)
        
        # Count every element in every file
        events = []
        for subdir in [proton_files, gustav_files, werner_files]:
            file_list = []
            for file in subdir:
                event_count = 0
                with gzip.open(file) as event_data:
                    for event in event_data:
                        event_count += 1
                file_list.append([file, event_count])
            events.append(file_list)
        
        data = []
        for elem in events:
            for i in elem:
                data.append(i)
                
        # Save metadata to a df
        df = pd.DataFrame(data, columns=['File_name', 'Event_count'])
        df['Particle'] = df['File_name'].apply(lambda x: False if 'proton' in x else True)
        df.to_csv(file_path, encoding='utf-8', index=False)
        
    return df