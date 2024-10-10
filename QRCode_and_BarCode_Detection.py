import cv2
import numpy as np
from pyzbar.pyzbar import decode

(width_c,height_c) = 1024,768

cap = cv2.VideoCapture(0)
cap.set(3,width_c)
cap.set(4,height_c)

while True:
    success,Original_img = cap.read()
    Original_img = cv2.resize(Original_img,(width_c,height_c))
    if not success:
        break
    else:
        for code in decode(Original_img):
            code_data = code.data.decode("utf-8")
            print(code_data)
            pts = np.array([code.polygon],np.int32)
            pts.reshape((-1,1,2))
            cv2.polylines(Original_img,[pts],True,(0,255,0),5)
            (x,y,w,h) = code.rect
            cv2.putText(Original_img,code_data,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
        cv2.imshow("Original Img",Original_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
