import numpy as np 
from pickle import dump
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle



if __name__ == "__main__":

    # load the face dataset
    data = np.load('5-celebrity-faces-embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)
    
    # shuffle the train set
    trainX, trainy = shuffle(trainX,trainy,random_state=42)

    # Create a SVC model
    model = SVC(kernel='linear', probability=True)

    # train the model
    model.fit(trainX,trainy)

    # save the model
    dump(model,open('SVC_model.pkl', 'wb'))

