from flask import Flask, render_template, Response, request, url_for, redirect
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
from flask_sqlalchemy import SQLAlchemy
from PIL import Image

global capture
capture=0

#Load pretrained face detection model    
net = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')
app.config['SECRET_KEY'] = 'mysecret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cobaaa.sqlite3'

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(1000))
    status = db.Column(db.String(1000))
    
db.create_all()

cam = cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    global capture
    while True:
        ret, img = cam.read()
        if ret:
            if(capture):
                capture=0
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = net.detectMultiScale(gray, 1.3, 5)

                for (x,y,w,h) in faces:
                    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                    
                #now = datetime.datetime.now()
                global User
                data1 = User.query.all()
                idd = []
                namee = []
                statuss = []
    
                for amounts in data1:
                    idd.append(amounts.id)
                    namee.append(amounts.name)
                    statuss.append(amounts.status)
                    
                aidi = idd[-1]
                p = os.path.sep.join(['dataset', "User." + str(aidi) + ".png"])
                try:
                    cv2.imwrite(p, gray[y:y+h,x:x+w])
                except:
                    cv2.imwrite(p, img)
                
            try:
                r, buffer = cv2.imencode('.jpg', cv2.flip(img,1))
                img = buffer.tobytes()
                yield (b'--img\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/')
def index():
    return render_template('form.html')

@app.route('/prosesForm', methods=['POST'])
def proses_task():
    name = request.form.get('name')
    status = request.form.get('status')
    new_User = User(name=name, status=status)
    db.session.add(new_User)
    db.session.commit()
    return redirect(url_for('camera'))

@app.route('/camera')
def camera():
    return render_template('cobaa.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=img')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        else:
            cam = cv2.VideoCapture(0)
           
    elif request.method=='GET':
        return render_template('cobaa.html')
    return render_template('cobaa.html')

#######################################################

global recognizer, detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

path = 'dataset'

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    
    global detector
    
    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

@app.route('/training')
def training():
    global recognizer, detector
    
    path = 'dataset'
    getImagesAndLabels(path)
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    recognizer.write('trainer/trainer.yml')
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run()
    
cam.release()
cv2.destroyAllWindows()     