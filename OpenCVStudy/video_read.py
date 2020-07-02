import cv2

# cap = cv2.VideoCapture(0) #创建视频对象
cap = cv2.VideoCapture("car.mp4")
while True:
    ret, frame = cap.read() #ret是否读成功,frame图像
    cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()