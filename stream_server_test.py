__author__ = 'zhengwang'

import numpy as np
import cv2
import socket
import io
stream = io.BytesIO()
font = cv2.FONT_HERSHEY_SIMPLEX


# Create a socket object
s = socket.socket()

# Define the port on which you want to connect
port = 9000

# connect to the server on local computer
#s.connect(('127.0.0.1', port))

# receive data from the server
#print s.recv(1024)
# close the connection
#s.close()

class VideoStreamingTest(object):
    def __init__(self):

        self.server_socket = socket.socket()
        self.server_socket.bind(('192.168.1.105', 8002))
        self.server_socket.listen(1)
        self.connection, self.client_address = self.server_socket.accept()
        self.connection = self.connection.makefile('rb')
        self.streaming()

    def streaming(self):

        try:

            print "Connection from: ", self.client_address
            print "Streaming..."
            print "Press 'q' to exit"


            stream_bytes = ' '
            while True:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    #image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_GRAYSCALE)
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

                    cv2.imshow('image', im)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.connection.close()
            self.server_socket.close()
            s.close()

if __name__ == '__main__':
    recognizer = cv2.createLBPHFaceRecognizer()
    recognizer.load('trainer.yml')
    print "classsifier loaded"
    VideoStreamingTest()
