from flask import Flask,render_template,url_for,Response
import cv2

app=Flask(__name__)
camera =cv2.VideoCapture(0)     #capturing video frames


def generate_frames():
    while True:
        success, frame = camera.read()               #read data from camera
        
        if not success:
            break
        else:
            #for face and eye detection.
            detector=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
            
            #detector.detectMultiScale() is a method that detects objects (faces,eyes in this case) in the input frame.
            faces=detector.detectMultiScale(frame,1.1,7)
            
            '''cv2.cvtColor() converts the BGR color image (standard OpenCV format) into a grayscale image.
                Grayscale images are easier to process for tasks like face detection.'''
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            #to draw the line around the face
            for(x,y,w,h) in faces:
                
                #cv2.rectangle() draws a blue rectangle around the detected face on the frame.
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
                
                #roi_gray represents the region of interest (ROI) in grayscale within the detected face area.
                roi_gray=gray[y:y+h, x:x+w]
                
                #roi_color represents the corresponding region in the original frame with color.
                roi_color=frame[y:y+h, x:x+w]
                
                #eye_cascade.detectMultiScale() searches for eyes within the grayscale face region (roi_gray).
                eyes=eye_cascade.detectMultiScale(roi_gray, 1.1 ,3)
                
                #For each detected eye within the face,cv2.rectangle() draws a green rectangle around each detected eye in the color region
                for(ex,ey,eh,ew) in eyes:
                    cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
                    
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
            
            

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
    

