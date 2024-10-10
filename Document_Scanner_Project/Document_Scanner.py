import cv2
import numpy as np
import os
from Utilities.Common_Utils import WarpImage,GetContours,EuclideanDistance,ReorderRectanglePoints
from colorama import Fore

(width_c,height_c) = 1000,750
w_p = int(210 * 2.25)
h_p = int(297 * 2.25)

cap = cv2.VideoCapture("http://192.168.1.35:8080/video")
cap.set(3,width_c)
cap.set(4,height_c)
cap.set(10,200)

pause_img = False
save_mode = ["BGR","RGB","HSV","Gray","Threshold","Threshold Inv"]
save_index = 0
total_saved_imgs = 0
color = (0,255,255)

def InitializeTrackbars(window_name,trackbar_name1,trackbar_name2,trackbar_name3):
    def Pass(x):
        pass
    cv2.namedWindow(window_name)
    cv2.resizeWindow(window_name,(360, 120))
    cv2.createTrackbar(trackbar_name1,window_name,0,255,Pass)
    cv2.createTrackbar(trackbar_name2,window_name,0,255,Pass)
    cv2.createTrackbar(trackbar_name3,window_name,1000,100000,Pass)

def Trackbars(window_name,trackbar_name1,trackbar_name2,trackbar_name3):
    trackbar1 = cv2.getTrackbarPos(trackbar_name1,window_name)
    trackbar2 = cv2.getTrackbarPos(trackbar_name2,window_name)
    trackbar3 = cv2.getTrackbarPos(trackbar_name3, window_name)
    return trackbar1,trackbar2,trackbar3

InitializeTrackbars("Trackbars","Canny Thr1","Canny Thr2","Min Area")

while True:
    success,Original_img = cap.read()
    Original_img = cv2.resize(Original_img,(width_c,height_c))
    Black_board = np.zeros_like(Original_img)
    if not success:
        break
    else:
        thr1,thr2,min_area = Trackbars("Trackbars","Canny Thr1","Canny Thr2","Min Area")
        contours = GetContours(Original_img,canny_thr=[thr1,thr2],min_area=min_area,filter_=4,show_canny=False,draw=False)
        if len(contours) != 0:
            biggest_contour = contours[0]
            if biggest_contour[0] == 4:
                p1,p2,p3,p4 = ReorderRectanglePoints(biggest_contour[2]).reshape((4,2))
                cv2.fillConvexPoly(Black_board,biggest_contour[2],color)
                cv2.polylines(Original_img,[biggest_contour[2]],True,color,5)
                if (EuclideanDistance(p1,p2) and EuclideanDistance(p3,p4)) > (EuclideanDistance(p1,p3) and EuclideanDistance(p2,p4)):
                    w_p = int(297 * 2.25)
                    h_p = int(210 * 2.25)
                elif (EuclideanDistance(p1,p2) and EuclideanDistance(p3,p4)) < (EuclideanDistance(p1,p3) and EuclideanDistance(p2,p4)):
                    w_p = int(210 * 2.25)
                    h_p = int(297 * 2.25)
                else:
                    w_p = int(297 * 2.25)
                    h_p = int(297 * 2.25)
                if not pause_img:
                    Warp_img = WarpImage(Original_img,biggest_contour[2],w_p,h_p,12,inverse=False)
                    Copy_Warp_img = Warp_img.copy()
                    cv2.imshow("Warp Img",Warp_img)
        cv2.putText(Original_img,f"Save Mode: {save_mode[save_index]}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)
        cv2.putText(Original_img,f"Pause Img: {pause_img}",(10,80),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)
        cv2.putText(Original_img,f"Total Saved Imgs: {total_saved_imgs}",(10,Original_img.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3)
        Result_img = cv2.addWeighted(Original_img,1,Black_board,0.3,0)
        cv2.imshow("Result Img",Result_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("a"):
        if save_index > 0:
            save_index -= 1
    elif key == ord("d"):
        if save_index < len(save_mode) - 1:
            save_index += 1
    elif key == ord("p"):
        pause_img = not pause_img
    elif key == ord("s"):
        with open("Count.txt","a+") as file:
            file.seek(0)
            content = file.readlines()
            if content:
                numbers = [int(line.strip()) for line in content]
                next_number = max(numbers) + 1
            else:
                next_number = 1
            file_path = os.path.join("Saved_Documents", f"Document_{next_number}.jpg")
            if save_mode[save_index] == "BGR":
                cv2.imwrite(file_path,Copy_Warp_img)
            elif save_mode[save_index] == "Gray":
                Gray_Warp_img = cv2.cvtColor(Copy_Warp_img,cv2.COLOR_BGR2GRAY)
                cv2.imwrite(file_path,Gray_Warp_img)
            elif save_mode[save_index] == "Threshold":
                Gray_Warp_img = cv2.cvtColor(Copy_Warp_img, cv2.COLOR_BGR2GRAY)
                Threshold_Warp_img = cv2.adaptiveThreshold(Gray_Warp_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                                           cv2.THRESH_BINARY,7,2)
                Threshold_Warp_img = cv2.medianBlur(Threshold_Warp_img,5)
                cv2.imwrite(file_path,Threshold_Warp_img)
            elif save_mode[save_index] == "Threshold Inv":
                Gray_Warp_img = cv2.cvtColor(Copy_Warp_img,cv2.COLOR_BGR2GRAY)
                Threshold_Warp_img_inv = cv2.adaptiveThreshold(Gray_Warp_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,
                                                               cv2.THRESH_BINARY_INV,7,2)
                cv2.medianBlur(Threshold_Warp_img_inv,5)
                cv2.imwrite(file_path,Threshold_Warp_img_inv)
            elif save_mode[save_index] == "RGB":
                RGB_Warp_img = cv2.cvtColor(Copy_Warp_img,cv2.COLOR_BGR2RGB)
                cv2.imwrite(file_path,RGB_Warp_img)
            elif save_mode[save_index] == "HSV":
                HSV_Warp_img = cv2.cvtColor(Copy_Warp_img,cv2.COLOR_BGR2HSV)
                cv2.imwrite(file_path,HSV_Warp_img)
            print(Fore.LIGHTGREEN_EX + f"[{save_mode[save_index]}] {file_path} saved.")
            file.write(f"{next_number}\n")
            total_saved_imgs += 1
            color = (0,255,0)
    elif key == ord("r"):
        with open("Count.txt","w") as file:
            file.write("")
            for i,files in enumerate(os.listdir("Saved_Documents")):
                files_dir = os.path.join("Saved_Documents",files)
                os.remove(files_dir)
                print(Fore.LIGHTRED_EX + f"[{i + 1}] {files_dir} removed.")
            break
    if key != ord("s"):
        color = (0,255,255)
cap.release()
cv2.destroyAllWindows()
