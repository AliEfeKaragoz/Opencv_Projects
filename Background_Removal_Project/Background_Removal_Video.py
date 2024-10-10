import cv2
import numpy as np
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

list_vid = os.listdir("Background_Videos")
vid_list = [os.path.join("Background_Videos",vid) for vid in list_vid]

index_img = 0

(width_c,height_c) = 700,500

cap = cv2.VideoCapture(0)
cap_2 = cv2.VideoCapture(vid_list[index_img])

segmentor = SelfiSegmentation()

while True:
    success,Original_img = cap.read()
    success_2,Img_BG = cap_2.read()
    Original_img = cv2.resize(Original_img,(width_c,height_c))
    if not success:
        break
    if not success_2:
        cap_2.set(cv2.CAP_PROP_POS_FRAMES,0)
        continue
    else:
        Img_BG = cv2.resize(Img_BG, (width_c, height_c))
        Img_BG = segmentor.removeBG(Original_img, Img_BG,0.8)
        Img_stacked = np.hstack((Original_img,Img_BG))

        cv2.imshow("Img Stacked",Img_stacked)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("a"):
        if index_img > 0:
            index_img -= 1
            cap_2.release()
            cap_2 = cv2.VideoCapture(vid_list[index_img])
    elif key == ord("d"):
        if index_img < len(list_vid) - 1:
            index_img += 1
            cap_2.release()
            cap_2 = cv2.VideoCapture(vid_list[index_img])
    elif key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
