#on mndy emotn trying to include


import threading
import SocketServer

import cv2
import numpy as np
import math
import io
stream = io.BytesIO()
font = cv2.FONT_HERSHEY_SIMPLEX
recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trainer.yml')
print "reconiszer loaded"

# distance data measured by ultrasonic sensor
sensor_data = " "





class SensorDataHandler(SocketServer.BaseRequestHandler):

    data = " "

    def handle(self):
        global sensor_data
        try:
            while self.data:
                self.data = self.request.recv(1024)
                sensor_data = round(float(self.data), 1)
                #print "{} sent:".format(self.client_address[0])
                print sensor_data
        finally:
            print "Connection closed on thread 2"


class VideoStreamHandler(SocketServer.StreamRequestHandler):


    # cascade classifiers


    def handle(self):

        global sensor_data
        stream_bytes = ' '


        # stream video frames one by one
        try:
            while True:

            #print "classsifier loaded"

                stream_bytes += self.rfile.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]

                    im = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
                    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
                    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

                        #Look for faces in the image using the loaded cascade file
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

                    print "Found "+str(len(faces))+" face(s)"



                        #Draw a rectangle around every found face
                    for (x,y,w,h) in faces:

                        image = cv2.resize(gray[y:y+h,x:x+w],
                    				(92,112),
                                    interpolation=cv2.INTER_LANCZOS4)

                            # Create rectangle around the face
                        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

                            # Recognize the face belongs to which ID
                        Id = recognizer.predict(image)



                            # Put text describe who is in the picture
                        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
                        cv2.putText(im, str(Id), (x,y-40), font, 2, (255,255,255), 3)
                        print str(Id)

                        #Save the result image


                #    SensorDataHandler()


                    cv2.imshow('image', im)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.rc_car.stop()
                        break

            cv2.destroyAllWindows()

        finally:
            print "Connection closed on thread 1"


class ThreadServer(object):

    def server_thread(host, port):
        server = SocketServer.TCPServer((host, port), VideoStreamHandler)
        server.serve_forever()

    def server_thread2(host, port):
        server = SocketServer.TCPServer((host, port), SensorDataHandler)
        server.serve_forever()

    distance_thread = threading.Thread(target=server_thread2, args=('192.168.1.105', 6003))
    distance_thread.start()
    video_thread = threading.Thread(target=server_thread('192.168.1.105', 6004))
    video_thread.start()

if __name__ == '__main__':

    ThreadServer()
