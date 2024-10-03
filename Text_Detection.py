import cv2
import pytesseract as pyt

pyt.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

(width_c,height_c) = 1000,750

cap = cv2.VideoCapture(0)
cap.set(3,width_c)
cap.set(4,height_c)

pause = False

while True:
    success,Original_img = cap.read()
    Original_img = cv2.resize(Original_img,(width_c,height_c))
    if not success:
        break
    else:
        Gray_img = cv2.cvtColor(Original_img,cv2.COLOR_BGR2GRAY)
        datas = pyt.image_to_data(Gray_img)
        for i,d in enumerate(datas.splitlines()):
            if i != 0:
                d = d.split()
                if len(d) == 12:
                    print(d[11])
                    x,y,w,h = int(d[6]),int(d[7]),int(d[8]),int(d[9])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.rectangle(Original_img,(x,y),(x+w,y+h),(0,0,255),2)
                    cv2.putText(Original_img,d[11],(x,y),font,0.7,(0,0,255),2)
        if not pause:
            cv2.imshow("Original Img",Original_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        pause = not pause
cap.release()
cv2.destroyAllWindows()
