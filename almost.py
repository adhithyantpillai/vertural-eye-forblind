import cv2
import cv2.cv as cv
import numpy as np
import os
import time
i_no=0
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
 
if __name__ == '__main__':
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
print 'please take an image to test the program'
time.sleep(3)
test=np.zeros((1,1125),np.float32)
cv.NamedWindow("camera", 1)
capture = cv2.VideoCapture(0)
num = 0
while True:
    ret,img = capture.read()
    cv2.imshow("camera", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
           print 'Capturing'
           cv2.imwrite("test.jpg",img)
           break
capture.release()
cv2.destroyAllWindows()
facechop("test.jpg")
dctim('fn3')
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
train=np.zeros((83,1),np.float32)
for g in range(83):
        if(g)> 41:
            train[g,0]=1
svm=cv2.SVM()
svm.train(dctvariable1,train,params=svm_params)
result=svm.predict(test)
if(result==1.0):
    print 'neutral'
    os.system("espeak 'neutral'")
    
if(result==0.0):
    print 'happy'
    os.system("espeak 'happy'")
    
    
#print img.shape




