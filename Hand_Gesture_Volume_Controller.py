import cv2
from cvzone.HandTrackingModule import HandDetector
from Utilities import Drawing_Utils as du
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

(width_c,height_c) = 1000,750

cap = cv2.VideoCapture(0)
cap.set(3,width_c)
cap.set(4,height_c)

detector = HandDetector(maxHands=1)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
"""volume.GetMute()
volume.SetMasterVolumeLevel(0.0,None)
volume.GetMasterVolumeLevel()"""
vol = volume.GetVolumeRange()
vol_bar = 10
vol_per = volume.SetMasterVolumeLevel(vol[0],None)

while True:
    success,Original_img = cap.read()
    Original_img = cv2.resize(Original_img,(width_c,height_c))
    if not success:
        break
    else:
        hands,Original_img = detector.findHands(Original_img,draw=False)
        if hands:
            color = (255,0,255)
            hand = hands[0]
            x1,y1 = hand["lmList"][4][:2]
            x2,y2 = hand["lmList"][8][:2]
            cx,cy = (x2 + x1) // 2,(y2 + y1) // 2
            length = math.hypot(x2 - x1,y2 - y1)
            if length <= 40 or length >= 300:
                color = (0,255,0)
            vol_bar = np.interp(length,[40,300],[10,310])
            vol_per = np.interp(length,[40,300],[0,100])
            smoothness = 1
            vol_per = smoothness * round(vol_per / smoothness)
            if vol_per > 100:
                vol_per = 100
            volume.SetMasterVolumeLevelScalar(vol_per/100,None)
            du.LineSegment(Original_img, (x1, y1), (x2, y2), (255, 0, 255), 4)
            cv2.circle(Original_img, (cx, cy), int(2.25 * 4), color, -1)
        cv2.rectangle(Original_img,(10, 10), (310, 60), (255, 0, 0), 3)
        cv2.rectangle(Original_img,(10,10),(int(vol_bar),60), (255,0,0), -1)
        cv2.putText(Original_img,f"{str(int(vol_per))}%",(320,45),cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,0,0),3)
        cv2.imshow("Original Img",Original_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
