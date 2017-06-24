
# coding: utf-8

# # Training MNIST with ordered and randomized order of pixels to a neural net

# The set contains 42.000 handwritten digits. The pictures cover 28x28 pixel and are stored in a 784 separate columns.

# In[ ]:

# Import in order of usage
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense

import csv


# Function to shuffle the columns of the DataFrame (shuffle the pixels order)
def shuffleDataFrame(df, shuffle=False):
    '''
    Input: 
    df = pd.DataFrame
    shuffle = Boolean
    
    Output:
    df with shuffles (True) or unshuffled (False) column-order    
    '''
    if shuffle:
        col = df.columns.tolist()
        random.shuffle(col)
        return df[col]
    else: return df



def main(df, shuffle=False, plots=True):
    # Decide whether the columns should be shuffled
    df = shuffleDataFrame(df, shuffle)

    # Splitting all data into training- and testing-set
    df_train ,df_test = train_test_split(df, test_size=0.3)

    # Preprocessing the data to fit into the neuronal net
    X_train = df_train.drop(['label'], axis=1).values
    y_train = np_utils.to_categorical(df_train['label'].values, 10)
    
    X_test = df_test.drop(['label'], axis=1).values
    y_test = np_utils.to_categorical(df_test['label'].values, 10)
    y_test_class = df_test['label'].values

    # Can be commented out be setting plots=False in main()
    if plots:
        # Checking the different shapes
        print('Shape Training-Data', X_train.shape)
        print('Shape Training-Label', y_train.shape)
        print('Shape Testing-Data', X_test.shape)
        print('Shape Training-Label', y_test.shape, '\n')
         
            
        # Visualizing the input vector. The vector has been reshaped for visual purposes.
        # The pixels are in a series, so a simple np.reshape([28,28]) will reset it to it's original image.
        fig = plt.figure(figsize=(12, 10))
    
        for i in range(16):
            picture = X_train[i].reshape([28,28])
            plt.subplot(4, 4, i+1).imshow(picture, cmap='gray', interpolation='none')
            plt.title("Class {}".format(df_train['label'].values[i]))
            plt.axis('off')

        plt.suptitle('Image examples of the input-vector with their target class label')
        plt.show()
        
        
    #The neural net will have 3 hidden layers with 300 nodes each.
    #It will have 784 input values and every node will be connected to every node of the previous and following layer.
    
    # Create the model: model
    model = Sequential()

    # Add the first hidden layer
    model.add(Dense(300, activation='relu', input_shape=(784,)))

    # Add the second hidden layer
    model.add(Dense(300, activation='relu'))

    # Add the third hidden layer
    model.add(Dense(300, activation='relu'))

    # Add the output layer
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train/255, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.3)

    # Test the model explicit with the test data
    score = model.evaluate(X_test, y_test, verbose=0)
    if plots:
        print('\nTest score:', score[0])
        print('Test accuracy:', score[1])
        
        
    # Saving the score and accuracy to a csv-file
    if shuffle:
        file_name = 'unshuffled.csv'
    else:
        file_name = 'shuffled.csv'
    
    with open(file_name, 'a', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([score[0], score[1]])  
        
           
        
    
    if plots:
        # Predicting the classes of the testing-set
        predicted_classes = model.predict_classes(X_test)

        # Check which items we got right / wrong
        correct_indices = np.nonzero(predicted_classes == y_test_class)[0]
        incorrect_indices = np.nonzero(predicted_classes != y_test_class)[0]
        
        
        # Visualizing some testing-data. Correctly and incorrectly predicted classes are contained.

        plt.figure(figsize=[9,9])
        for i, correct in enumerate(correct_indices[:9]):
            plt.subplot(3,3,i+1)
            plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
            plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test_class[correct]))
            plt.axis('off')
            plt.suptitle('Image examples of correct predictions')
        plt.show()
    
        plt.figure(figsize=[9,9])
        for i, incorrect in enumerate(incorrect_indices[:9]):
            plt.subplot(3,3,i+1)
            plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
            plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test_class[incorrect]))
            plt.axis('off')
            plt.suptitle('Image examples of incorrect predictions')
    plt.show()
    
    


# In[ ]:

# Loading all data in a single DataFrame
df = pd.read_csv('mnist.csv')

with open('shuffled.csv', 'w') as file:
    pass

with open('unshuffled.csv', 'w') as file:
    pass

for _ in range(100):
    #main(df, shuffle=False, plots=True)
    main(df, shuffle=False, plots=False)

    main(df, shuffle=True, plots=False)


# To compare both runs the mean and standard deviation of every neural nets accuracy will be computed.

# In[ ]:

shuffled = pd.read_csv('shuffled.csv', header=None)
unshuffled = pd.read_csv('unshuffled.csv', header=None)

print('Shuffled: {} +/- {} %'.format(shuffled[1].mean()*100, shuffled[1].std()*100))
print('Unshuffled: {} +/- {} %'.format(unshuffled[1].mean()*100, unshuffled[1].std()*100))


# In[ ]:



