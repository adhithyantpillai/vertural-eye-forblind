#on mndy emotn trying to include


import threading
import SocketServer

import cv2
import numpy as np
import math
import io
import cv2.cv as cv
i_no=0
stream = io.BytesIO()
font = cv2.FONT_HERSHEY_SIMPLEX
recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trainer.yml')
print "reconiszer loaded"

# distance data measured by ultrasonic sensor
sensor_data = " "






def facechop(image):
    global i_no
    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)

    minisize = (img.shape[1],img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        i_no=i_no+1
        face_file_name = "face.jpg"
        #face_file_name='E:\\Emotion_recognition\\faces\\'+str(i_no)+'.jpg'
        newimage = cv2.resize(sub_face,(240,240))
        cv2.imwrite(face_file_name, newimage)

        #cv2.imshow(image, img)

    return
def dctim(imge):
            B=16 #blocksize
            fn3= 'face.jpg'
            img1 = cv2.imread(fn3, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            h,w=np.array(img1.shape[:2])/B * B
            print h
            print w
            img1=img1[:h,:w]
            blocksV=h/B
            blocksH=w/B
            vis0 = np.zeros((h,w), np.float32)
            Trans = np.zeros((h,w), np.float32)

            a=np.zeros((16,16),np.float32)
            vis0[:h, :w] = img1
            for row in range(blocksV):
                for col in range(blocksH):
                    currentblock = cv2.dct(vis0[row*B:(row+1)*B,col*B:(col+1)*B])
                    Trans[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock

            cv2.cv.SaveImage('Transformed.jpg', cv2.cv.fromarray(Trans))
            img=cv2.imread('Transformed.jpg',-1)
            #cv2.imshow('dct',img)
            return

#emotion
test=np.zeros((1,1125),np.float32)

svm_params = dict( kernel_type = cv2.SVM_LINEAR,svm_type=cv2.SVM_C_SVC,C=2.67, gamma=5.383 )
dctvariable1=np.zeros((83,1125),np.float32)
ext='.jpg'
for image_no in range(83):
    image='face_extracted/'+str(image_no+1)+ext
    facechop(image)
    dctim('fn3')
#del(capture)
    img=cv2.imread('Transformed.jpg',-1)
        #cv2.waitKey(0)
    k=0

    for i in range(15):

        for j in range(15):

            a=img[i*16:((i+1)*16-1),(j)*16:((j+1)*16)-1]
            dctvariable1[image_no,k]=a[0,1]
            k=k+1
            dctvariable1[image_no,k]=a[1,0]
            k=k+1
            dctvariable1[image_no,k]=a[2,0]
            k=k+1
            dctvariable1[image_no,k]=a[1,1]
            k=k+1
            dctvariable1[image_no,k]=a[0,2]
            k=k+1

train=np.zeros((83,1),np.float32)
for g in range(83):
        if(g)> 41:
            train[g,0]=1
svm=cv2.SVM()
svm.train(dctvariable1,train,params=svm_params)

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

                        imagee = cv2.resize(gray[y:y+h,x:x+w],
                    				(92,112),
                                    interpolation=cv2.INTER_LANCZOS4)

                            # Create rectangle around the face
                        cv2.imwrite("test.jpg",imagee)
                        facechop("test.jpg")
                        dctim('fn3')
                        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

                            # Recognize the face belongs to which ID
                        Id = recognizer.predict(imagee)
                        img=cv2.imread('Transformed.jpg',-1)
                        k=0
                        for i in range(15):

                            for j in range(15):

                                    a=img[i*16:((i+1)*16-1),(j)*16:((j+1)*16)-1]
                                    test[0,k]=a[0,1]
                                    k=k+1
                                    test[0,k]=a[1,0]
                                    k=k+1
                                    test[0,k]=a[2,0]
                                    k=k+1
                                    test[0,k]=a[1,1]
                                    k=k+1
                                    test[0,k]=a[0,2]
                                    k=k+1

                        result=svm.predict(test)



                            # Put text describe who is in the picture
                        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
                        cv2.putText(im, str(Id), (x,y-40), font, 2, (255,255,255), 3)
                        print str(Id)
                        if(result==1.0):
                            print 'neutral'
                            #os.system("espeak 'neutral'")

                        if(result==0.0):
                            print 'happy'
                            #os.system("espeak 'happy'")


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

    distance_thread = threading.Thread(target=server_thread2, args=('192.168.1.101', 6002))
    distance_thread.start()
    video_thread = threading.Thread(target=server_thread('192.168.1.101', 6003))
    video_thread.start()

if __name__ == '__main__':
    print "main enter"
    ThreadServer()
