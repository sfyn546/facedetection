import cv2
face=cv2.CascadeClassifier("haarfiles/haarcascade_frontalface_default.xml")

#cap=cv2.VideoCapture("0")
cap=cv2.VideoCapture(0) #cam fetch
cap.set(3,640) #width
cap.set(4,480)
cap.set(10,50)



while True:
    ret,img=cap.read()
    if(ret):
        imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face.detectMultiScale(imgGray,1.1,4)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow("Video",img)
    if(cv2.waitKey(1)==ord('q')):
        break



