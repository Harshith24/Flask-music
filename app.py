from flask import Flask, request, redirect, url_for, render_template, Response
from camera import VideoCamera

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

def gen(camera):
    while True:
        global df1
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

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