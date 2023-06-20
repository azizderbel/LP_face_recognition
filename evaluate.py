import numpy as np 
from pickle import load
from sklearn.metrics import accuracy_score



if __name__ == "__main__":

    # load the face dataset
    data = np.load('5-celebrity-faces-embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

    # load the trained model
    model = load(open('SVC_model.pkl', 'rb'))

    y_pred_train = model.predict(trainX)
    y_pred_test = model.predict(testX)

    print(trainX[0])
    print(y_pred_test)

    # calculate the accuracy score
    train_acc = accuracy_score(trainy, y_pred_train)
    test_acc = accuracy_score(testy, y_pred_test)

    print(f"The accuracy on the train set is {train_acc}")
    print(f"The accuracy on the test set is {test_acc}")
