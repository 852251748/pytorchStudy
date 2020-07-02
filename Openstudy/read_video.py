import numpy as np
import cv2
# cap = cv2.VideoCapture("玩滑板的妹子.mp4")
cap = cv2.VideoCapture("rtsp://113.136.42.39:554/PLTV/88888888/224/3221226107/10000100000000060000000001759238_0.smil")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ret =", ret)
        break
    cv2.imshow("frame", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
