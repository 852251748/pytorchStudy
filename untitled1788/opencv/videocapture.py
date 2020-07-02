import cv2 as cv

cap = cv.VideoCapture("D:\pycharmworkspace\Openstudy\玩滑板的妹子.mp4")
while True:
    ret, img = cap.read()
    cv.imshow("frame", img)
    if cv.waitKey(30) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
