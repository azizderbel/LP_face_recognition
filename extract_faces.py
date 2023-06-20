import numpy as np 
import cv2 
import os

# Extract faces from images to a 160 * 160 pixals shapes
def extract_face(filename,face_detector,required_size=(160, 160)):
    # load image from file
    image = cv2.imread(filename=filename)
    pixels = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    faces = face_detector.detectMultiScale(pixels, scaleFactor=1.01, minNeighbors=5)
    if len(faces) > 0:
            x1, y1, width, height = faces[0]
    # deal with negative pixel index
            x1, y1 = abs(x1),abs(y1)
            x2, y2 = x1 + width, y1 + height
    # extract the face
            face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
            face = cv2.resize(face,required_size)
            return face

def load_faces(dir,face_detector):
    faces = list()
    # enumerate files
    for filename in os.listdir(dir):
        path = dir + filename
        face = extract_face(path,face_detector)
        if type(face) != type(None):
            faces.append(face)
    return faces

def load_dataset(dir,face_detector):
    # list for faces and labels
    X, y = list(), list()
    for subdir in os.listdir(dir):
        path = dir + subdir + '/'
        faces = load_faces(path,face_detector)
        labels = [subdir for i in range(len(faces))]
        print("loaded %d sample for class: %s" % (len(faces),subdir)) # print progress
        X.extend(faces)
        y.extend(labels)
    return np.asarray(X), np.asarray(y)


if __name__ == '__main__':

    # Load the face detector
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # load train dataset
    trainX, trainy = load_dataset('data/train/',face_detector)
    print(trainX.shape, trainy.shape)
    # load test dataset
    testX, testy = load_dataset('data/val/',face_detector)
    print(testX.shape, testy.shape)
    # save and compress the dataset for further use
    np.savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)