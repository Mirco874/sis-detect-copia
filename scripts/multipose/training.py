import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import time


DATA_PATH = os.path.join('TRAIN_DATA'); 
detections = np.array(['caminando', 'corriendo','peleando','amenazando'])

number_of_train_videos = 30
number_of_frames_per_video = 35

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    print(len(keypoints_with_scores)) 
    
    for person in keypoints_with_scores:
        print("---------------------------")
        print(person)
        print("---------------------------") 
        #land_mark = np.array([person.x, person.y, person.z, person.visibility]).flatten();
        #pose_landmarks.append(land_mark); 
        #print(len(pose_landmarks), "landmarks") 
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

    return [];


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

cap = cv2.VideoCapture(0)

for detection in detections:
    ##input("recording: "+detection);
    
    for video_index in range(0,number_of_train_videos):
        for frame_index in range(0,number_of_frames_per_video):
            print(frame_index, video_index)
            ret, frame_image = cap.read();
           
            # # Resize image
            # img = frame_image.copy()
            # img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256,256)
            # input_img = tf.cast(img, dtype=tf.int32)
    
            # # Detection section
            # results = movenet(input_img)
            # keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
            
            # print("###########################")
            # print(keypoints_with_scores)
            # print("###########################")

            if frame_index == 0: 
                cv2.putText(frame_image, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame_image, 'Collecting frames for {} Video Number {}'.format(detection, video_index), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.putText(frame_image, 'Collecting frames for {} Video Number {} Frame {}'.format(detection, video_index, frame_index), (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            #     input("presiona para iniciar")
            #     if(video == 14):
            #         print("mitad")

            # ##pose land marks of frame 0,1,2,3,4,...
            # # Render and save keypoints 
            # else:
            #     pose_landmarks = loop_through_people(frame_image, keypoints_with_scores, EDGES, 0.2)
            #     npy_path = os.path.join(DATA_PATH, detection, str(video), str(frame));
            #     ##np.save(npy_path, pose_landmarks)
            ###loop_through_people(frame_image, keypoints_with_scores, EDGES, 0.2) 
            time.sleep(2)           
            cv2.imshow('Movenet Multipose', frame_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# while cap.isOpened():
#     ret, frame = cap.read()
    
#     # Resize image
#     img = frame.copy()
#     img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256,256)
#     input_img = tf.cast(img, dtype=tf.int32)
    
#     # Detection section
#     results = movenet(input_img)
#     keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    
#     # Render keypoints 
#     loop_through_people(frame, keypoints_with_scores, EDGES, 0.2)
    
#     cv2.imshow('Movenet Multipose', frame)
    
#     if cv2.waitKey(10) & 0xFF==ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()