import numpy as np 
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from pickle import dump
from sklearn.preprocessing import MinMaxScaler
from keras.applications.vgg16 import VGG16,preprocess_input


def get_embedding(model, face):
    """# scale pixel values
    face = face.astype('float32') # 160 * 160 * 3 image
    # standardization
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    # transfer face into one sample (3 dimension to 4 dimension)
    sample = np.expand_dims(face, axis=0) 
    # make prediction to get embedding
    yhat = model.embeddings(sample)
    return yhat[0] # 1 * 1 * 512"""
    face = face.astype('float32')
    face = np.expand_dims(face, axis=0) 
    x = preprocess_input(face)
    features = model.predict(x)
    return features.ravel()
    

def normalize_embeddings(vector):
    in_encoder = MinMaxScaler(feature_range=(0,1))
    return in_encoder.fit_transform(vector)

def encode_target(target):
    encoder = LabelEncoder()
    target = encoder.fit_transform(target)
    return encoder,target


if __name__ == "__main__":

    # load the face dataset
    data = np.load('5-celebrity-faces-dataset.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Loaded: ', trainX.shape, trainy.shape, testX.shape, testy.shape)

    # load the facenet model
    facenet_model = VGG16(weights = 'imagenet', include_top=False)

    # convert each face in the train set into embedding
    emdTrainX = list()
    for face in trainX:
        emd = get_embedding(facenet_model, face)
        emdTrainX.append(emd)
    emdTrainX = np.asarray(emdTrainX)
    emdTrainX = normalize_embeddings(emdTrainX)
    print(emdTrainX.shape)

    # convert each face in the test set into embedding
    emdTestX = list()
    for face in testX:
        emd = get_embedding(facenet_model, face)
        emdTestX.append(emd)
    emdTestX = np.asarray(emdTestX)
    emdTestX = normalize_embeddings(emdTestX)
    print(emdTestX.shape)
    

    # label encoder
    train_encoder,trainy = encode_target(target=trainy)
    test_encoder,testy = encode_target(target=testy)

    dump(train_encoder,open('encoder.pkl', 'wb'))

    # save arrays to one file in compressed format
    np.savez_compressed('5-celebrity-faces-embeddings.npz', emdTrainX, trainy, emdTestX, testy)