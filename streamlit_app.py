import streamlit as st
from keras_facenet import FaceNet
from embeddings import get_embedding,normalize_embeddings
import numpy as np
import cv2
from pickle import load


def clear_flag():
    if 'flag' in st.session_state:
        del st.session_state['flag']

def set_flag():
    st.session_state['flag'] = True

def preprocess(image):
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(image, scaleFactor=1.01, minNeighbors=5)
    if len(faces) > 0:
            x1, y1, width, height = faces[0]
            x1, y1 = abs(x1),abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = image[y1:y2, x1:x2]
            face = cv2.resize(face,(160,160))
            # start the embedding process
            face_net_model = FaceNet()
            embedded_image = get_embedding(model=face_net_model,face=face)
            embedded_image = normalize_embeddings(np.reshape(embedded_image,(-1,1)))
            return face,np.reshape(embedded_image,(1,-1))

    
def main():

    st.title("Face recognition system")
    clf = load(open('SVC_model.pkl', 'rb'))
    encoder = load(open('encoder.pkl', 'rb'))

    uploaded_file = st.file_uploader("Choose a file",on_change=set_flag)

    if uploaded_file:
        bytes_data = uploaded_file.read()
        image_array = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), -1)
        image_array = cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)
        st.image(image_array)

        if 'flag' in st.session_state:
            with st.spinner('The image preprocessing is going on'):
                st.session_state.output = preprocess(image_array)
            
        if st.session_state.output:
            _ , embedded_image = st.session_state.output
            pred_btn = st.button(label='Predict its name !', on_click = clear_flag)
            if pred_btn:
                y_pred = clf.predict(embedded_image)
                person_name = encoder.inverse_transform(y_pred)[0]
                st.write("this person is " + person_name)


if __name__ == '__main__':
    main()