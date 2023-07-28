import cv2


name = "peleaizq.avi"
cap = cv2.VideoCapture(1)

fourcc = cv2.VideoWriter_fourcc(*'XVID');
out = cv2.VideoWriter(name, fourcc, 20.0, (640,480))



while(True):
    ret, frame = cap.read();
    if(ret == True):
        #frame = cv2.flip(frame,0)
        out.write(frame);
        cv2.imshow("frame",frame);
        if(cv2.waitKey(1) & 0xFF == ord("q")):
            break;
        
    else: 
        break;

cap.release();
out.release()
cv2.destroyAllWindows();        






