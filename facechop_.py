import cv2
import cv2.cv as cv
##import serial
image='28.jpg'
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
  face_file_name = "img28.jpg"
  newimage = cv2.resize(sub_face,(112,92))
  cv2.imwrite(face_file_name, newimage)
cv2.imshow(image, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
