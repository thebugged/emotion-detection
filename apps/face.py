import cv2
import tempfile
import numpy as np
import tensorflow as tf
import streamlit as st

from PIL import Image
from streamlit_option_menu import option_menu
from keras.preprocessing.image import img_to_array

def face_page():
    saved_model_path = 'models/fmodel.json'
    saved_weights_path = 'models/fmodel_weights.h5'

    with open(saved_model_path, 'r') as json_file:
        json_savedModel = json_file.read()

    model = tf.keras.models.model_from_json(json_savedModel)
    model.load_weights(saved_weights_path)
    # model = load_model('models/fmodel.h5')

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


    st.caption("switch between facial & audio detection from sidebar ⬅️")
    selected_page = option_menu(
            menu_title = None,
            options = ["Image", "Video",
                       "WebCam"
                       ],
            icons=['card-image', 'camera-video',
                  'person-video'
                   ],
            orientation="horizontal",
        )

    face_classifier = cv2.CascadeClassifier('misc/haarcascade_frontalface_default.xml')


    if selected_page == "Image":
        st.caption("sample [images](https://unsplash.com/s/photos/happy-face)")
        img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'])
        image = None 

        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
        # else:
        #     st.warning('Please upload an image.')
        
        if image is not None: 
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)

                    prediction = model.predict(roi)[0]
                    label = emotion_labels[prediction.argmax()]

                    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)
                    cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            st.image(image, width=450)
        
    if selected_page == "Video":
        st.caption("sample [videos](https://www.pexels.com/video/roller-coaster-852415/)")
        video_file_buffer = st.file_uploader("Upload a video", type=["mp4", "mov", 'avi', 'asf', 'm4v'])
        stframe = st.empty()

        if video_file_buffer is not None:
            tfflie = tempfile.NamedTemporaryFile(delete=False)
            tfflie.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tfflie.name)

            codec = cv2.VideoWriter_fourcc('v', 'p', '0', '9')
            out = cv2.VideoWriter('misc/output2.mp4', codec, 30, (640, 480))

            while vid.isOpened():
                ret, frame = vid.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray)

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                    if np.sum([roi_gray]) != 0:
                        roi = roi_gray.astype('float') / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)

                        prediction = model.predict(roi)[0]
                        label = emotion_labels[prediction.argmax()]

                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 3)
                        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

                out.write(frame)

                frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                frame = cv2.resize(frame, (640, 480))
                stframe.image(frame, channels='BGR', use_column_width=True)


            vid.release()
            out.release()
        # else:
        #     st.warning('Please upload a video.')
        

    if selected_page == "WebCam":
        st.caption("The webcam feature only works when hosted from local machine(local host).")
        use_webcam = st.toggle('Use Webcam')
        stframe = st.empty()

        if use_webcam:
            vid = cv2.VideoCapture(0)
            codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            out = cv2.VideoWriter('misc/output1.mp4', codec, 30, (640, 480))

            while use_webcam:
                ret, frame = vid.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray)

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                    if np.sum([roi_gray]) != 0:
                        roi = roi_gray.astype('float') / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)

                        prediction = model.predict(roi)[0]
                        label = emotion_labels[prediction.argmax()]

                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                out.write(frame)

                frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                frame = cv2.resize(frame, (640, 480))
                stframe.image(frame, channels='BGR', use_column_width=True)


            vid.release()
            out.release()


if __name__ == "__main__":
    face_page()
