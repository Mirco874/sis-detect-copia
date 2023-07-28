import cv2 
cap = cv2.VideoCapture(1)

flag = 1
num = 1


while (cap.isOpened()):
    ret_flag, Vshow = cap.read();
    cv2.imshow("Capture_test", Vshow)
    k= cv2.waitKey(1) & 0xFF
    if k== ord("s"):
        cv2.imwrite("./pistola/"+str(num)+".jpg", Vshow);
        num+=1;
    elif(k==ord("q")):
        break;


cap.release();
cv2.destroyAllWindows();


