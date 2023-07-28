import cv2
import mediapipe as mp
import time
import numpy as np
import os

DATA_PATH = os.path.join('TRAIN_DATA'); 
detections = np.array(['asalto_pistola', 'asalto_cuchillo']) 
number_of_train_videos = 40
number_of_frames_per_video = 35

pose_detector_model = mp.solutions.pose
pose = pose_detector_model.Pose();

cap = cv2.VideoCapture(1);
prediction_time = 0;


for detection in detections:
    input(detection)
    for video in range(number_of_train_videos):
        for frame in range(number_of_frames_per_video):
            sucesss, image = cap.read();
            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB);
            results = pose.process(imgRGB);

            if(results.pose_landmarks):
               
                mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, pose_detector_model.POSE_CONNECTIONS)
                pose_landmarks = [];
                for result in results.pose_landmarks.landmark:
                    land_mark = np.array([result.x, result.y, result.z, result.visibility]).flatten();
                    pose_landmarks.append(land_mark);

                if frame == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(detection, video), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    input("presiona")
                    if(video == 10):
                        print("medio")
                    elif(video == 25):
                        print("esquina")
                    
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(detection, video), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                    npy_path = os.path.join(DATA_PATH, detection, str(video), str(frame));
                    np.save(npy_path, pose_landmarks)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
