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
        self.server_socket.bind(('192.168.1.101', 9000))
        self.server_socket.listen(1)
        self.connection, self.client_address = self.server_socket.accept()
        self.connection = self.connection.makefile('rb')
        self.streaming()

    def streaming(self):

        try:

            print "Connection from: ", self.client_address
            print "Streaming..."
            print "Press 'q' to exit"
            #template = cv2.imread('Socket.jpg')
            #template = cv2.resize(template, (0,0), fx=0.5, fy=0.5)
            #templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)


            stream_bytes = ' '
            while True:
                template = cv2.imread('Socket.jpg')
                template = cv2.resize(template, (0,0), fx=0.5, fy=0.5)
                templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find('\xff\xd8')
                last = stream_bytes.find('\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    #image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_GRAYSCALE)
                    im = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_UNCHANGED)
                    image = cv2.resize(im, (0,0), fx=0.5, fy=0.5)
                    #cv2.imshow('image', im)
                    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                #  templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Find template
                    result = cv2.matchTemplate(imageGray,templateGray, cv2.TM_CCOEFF)
                    
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    top_left = max_loc
                    h,w = templateGray.shape
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    cv2.rectangle(image,top_left, bottom_right,(0,0,255),4)

# Show result
                    #cv2.imshow("Template", template)
                    cv2.imshow("Result", image)

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
