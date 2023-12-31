import cv2
import matplotlib.pyplot as plt
kamera= cv2.VideoCapture(0)

while True:
    ret,kare=kamera.read()
                           
    

    face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye=cv2.CascadeClassifier("haarcascade_eye.xml")
    gray=cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray,1.1,4) 

    for (x,y,w,h,) in faces:
        yeni=kare[y:y+h//2,x:x+w]
        gray_new=cv2.cvtColor(yeni,cv2.COLOR_BGR2GRAY)
        eyes=eye.detectMultiScale(gray_new,1.1,4) 
        cv2.rectangle(kare,(x,y),(x+w,y+h),(255,0,0),2) 
        for (a,b,c,d) in eyes:
            cv2.rectangle(yeni,(a,b),(a+c,b+d),(0,255,0),2) 

    cv2.imshow("kamera",kare)

    if cv2.waitKey(25) & 0xFF == ord('q'): 
                                        
        break


kamera.release()
cv2.destroyAllWindows()
