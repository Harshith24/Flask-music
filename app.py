from flask import Flask, request, redirect, url_for, render_template, Response
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from musicSearch import *

app = Flask(__name__)

faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)
# clicked = False

#print('cv2.CAP_PROP_FRAME_WIDTH :', cv2.CAP_PROP_FRAME_WIDTH)   # 3
#print('cv2.CAP_PROP_FRAME_HEIGHT:', cv2.CAP_PROP_FRAME_HEIGHT)  # 4
#print('cv2.CAP_PROP_FPS         :', cv2.CAP_PROP_FPS)           # 5
#print('cv2.CAP_PROP_FRAME_COUNT :', cv2.CAP_PROP_FRAME_COUNT)   # 7


#model
model = Sequential()
# Convolutional Layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the data for Dense layers
model.add(Flatten())

# Dense Layers
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.load_weights('2emotion_recognition_model.h5')

show_text=[0]
emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}

# model.summary()

@app.route('/')
def hello_world():
    global clicked
    clicked = False
    return render_template('index.html')

def gen_frames():  
    global currentEmotion
    global clicked
    if not clicked:
        while True:
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else:
                #convert to gray scale because dataset is trained on grayscale images
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect the face in the webcam
                faces = faceCascade.detectMultiScale(gray_frame, 1.1, 5)

                for (x,y,w,h) in faces:
                    # To draw a rectangle around the detected face  
                    cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(0,255,255),2)

                    # extract that specific region so we can crop and pass it into the model
                    roi_gray_frame = gray_frame[y:y+h, x:x+w]	
                    # roi_gray_frame = gray[y:y + h, x:x + w]		    
                    # cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                    # cropped_img = cv2.cvtColor(roi_gray_frame, cv2.COLOR_BGR2GRAY)

                    # resize, normalize, expand to 3D, pass and get prediction from model. Update the emotion result.  
                    cropped_img = cv2.resize(roi_gray_frame, (48, 48))
                    # cropped_img = cropped_img / 255.0
                    cropped_img = np.expand_dims(cropped_img, axis=0)
                    result = model.predict(cropped_img)
                    maxindex = int(np.argmax(result))
                    emotion_label = emotion_dict[maxindex]
                    print(emotion_label)			    
                    show_text[0] = maxindex 
                    cv2.putText(gray_frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                
                # Render onto the flask page
                ret, buffer = cv2.imencode('.jpeg', gray_frame) 
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                currentEmotion = emotion_dict[maxindex]


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name

@app.route('/music')
def music_page():
    global clicked
    clicked = True
    print("cuurent emotion", currentEmotion)
    searchMus = SearchMusic()
    playlist_name, music_dict, spotify_url = searchMus.search(currentEmotion)
    return render_template('music.html', emotion=currentEmotion, playlistName = playlist_name, music=music_dict, playlistURL=spotify_url)

# @app.route('/login', methods=['POST', 'GET'])
# def login():
#     if request.method=="POST":
#         user = request.form["nm"]
#         return redirect(url_for('success', name=user))
    

if __name__ == "__main__":
    app.run()