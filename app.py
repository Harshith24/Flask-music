from flask import Flask, request, redirect, url_for, render_template, Response
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Input, Dense

app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Input layer
# print(camera.get(3), camera.get(4))
# keras_input_layer = Input(shape=(int(camera.get(3)), int(camera.get(4))))

# # First layer
# keras_layer_1 = Dense(64)(keras_input_layer)

# # Second layer
# keras_layer_2 = Dense(32)(keras_layer_1)

# #output
# keras_output_layer = Dense(2)(keras_layer_2)

#print('cv2.CAP_PROP_FRAME_WIDTH :', cv2.CAP_PROP_FRAME_WIDTH)   # 3
#print('cv2.CAP_PROP_FRAME_HEIGHT:', cv2.CAP_PROP_FRAME_HEIGHT)  # 4
#print('cv2.CAP_PROP_FPS         :', cv2.CAP_PROP_FPS)           # 5
#print('cv2.CAP_PROP_FRAME_COUNT :', cv2.CAP_PROP_FRAME_COUNT)   # 7


#model
model = Sequential()
model.add(Input(shape=(int(camera.get(4)), int(camera.get(3)), 3)))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(2))

# model.summary()

@app.route('/')
def hello_world():
    return render_template('index.html')

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            resized_frame = cv2.resize(frame, (640, 480))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            normalized_frame = resized_frame / 255.0
            input_data = np.expand_dims(normalized_frame, axis=0)
            # ppr_img = np.array(input_data)
            result = model(input_data)
            print(result)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/success/<name>')
def success(name):
    return 'welcome %s' % name

# @app.route('/login', methods=['POST', 'GET'])
# def login():
#     if request.method=="POST":
#         user = request.form["nm"]
#         return redirect(url_for('success', name=user))
    

if __name__ == "__main__":
    app.run()