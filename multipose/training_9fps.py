from pygame import mixer

#Instantiate mixer
mixer.init()

#Load audio file
mixer.music.load('sonido_alarma.mp3')

print("music started playing....")

#Set preferred volume
mixer.music.set_volume(0.2)

import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import time


DATA_PATH = os.path.join('TRAIN_DATA'); 
#detections = np.array(['caminando_izquierda', 'caminando_derecha', 'pelea_izquierda', 'pelea_derecha'])
#detections = np.array(['golpe_derecha_mirada_derecha', 'golpe_izquierda_mirada_derecha','golpe_derecha_mirada_izquierda', 'golpe_izquierda_mirada_izquierda'])
#  apuntando_derecha, apuntando_izquierda, caminando_derecha, caminando_izquierda
detections = np.array(['caminando_derecha'])
number_of_train_videos = 5
number_of_frames_per_video = 10
#borrar los primeros 2 de caminar derecha
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

# Dibujar los 17 landmarks en un segundo
#regresa los 13 puntos importantes
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    ## un arreglo de 17 landmarks, cada uno con 2 coordenadas y 1 score
    person1=(keypoints_with_scores[0])
    #print("///ANTES///")
    #print(person1)


    pose_data = np.array(person1[5:])

    # draw_connections(frame, person1, edges, confidence_threshold)
    normalized_points=draw_keypoints(frame, pose_data, confidence_threshold)
    #print("///DESPUES///")
    #print(normalized_points)

    ####pose_training_data = np.array(normalized_points[5:13])
    #print(pose_training_data);

    return normalized_points;


    #for person in keypoints_with_scores:
    #     #pose_data=draw_keypoints(frame, person, confidence_threshold)
         #draw_connections(frame, person, edges, confidence_threshold)
         #draw_keypoints(frame, person, confidence_threshold)
    #     # print("---------------------------")
    #     # print(type(pose_data))
    #     # if(pose_data is None):
    #     #     pose_data = np.zeros(51)
    #     # else:
            


    #     # print(pose_data)
    #     # print("---------------------------") 

    #     #land_mark = np.array([person.x, person.y, person.z, person.visibility]).flatten();
    #     #pose_landmarks.append(land_mark); 
    #     #print(len(pose_landmarks), "landmarks") 
        

    #return person1;

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)

#regresa los 17 landmarks normalizados
def draw_keypoints(frame, keypoints, confidence_threshold):
    filtered_keypoints = [];
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    ##para cada coordenada
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)
            filtered_keypoints.append(kp);
        else:
            filtered_keypoints.append(np.zeros(3))
    return(filtered_keypoints)




cap = cv2.VideoCapture(0)

golpe_derecha_mirada_derecha = [];
golpe_izquierda_mirada_derecha = [];
golpe_derecha_mirada_izquierda = [];
golpe_izquierda_mirada_izquierda = [];


apuntando_derecha = [];
apuntando_izquierda = [];
caminando_derecha = [];
caminando_izquierda = [];

for detection in detections:
    videos_list = [];
    for video_index in range(number_of_train_videos):
        video_frames_list = []; # todos los frames del video n
        for frame_index in range(number_of_frames_per_video):
            ret, frame = cap.read()
            img = frame.copy()
            img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256,256)
            input_img = tf.cast(img, dtype=tf.int32)
            results = movenet(input_img)


            keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
            training_points = loop_through_people(frame, keypoints_with_scores, EDGES, 0.15)

            if frame_index == 0: 
                cv2.putText(frame, 'STARTING COLLECTION', (120,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(detection, video_index+1), (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                #input("trabajo: " +detection+" numero: "+ str(video_index+1));
                mixer.music.play()
                cv2.waitKey(4500)
                mixer.music.stop()

            else: 
                cv2.putText(frame, 'Collecting frames for {} Video Number {}'.format(detection, video_index+1), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)

            video_frames_list.append(training_points);

            if cv2.waitKey(10) & 0xFF==ord('q'):
                break
        videos_list.append(video_frames_list)

    if(detection == "golpe_derecha_mirada_derecha"):
        golpe_derecha_mirada_derecha.append(videos_list)

    elif(detection == "golpe_izquierda_mirada_derecha"):
        golpe_izquierda_mirada_derecha.append(videos_list)

    elif(detection == "golpe_derecha_mirada_izquierda"):
        golpe_derecha_mirada_izquierda.append(videos_list)
    
    elif(detection == "golpe_izquierda_mirada_izquierda"):
        golpe_izquierda_mirada_izquierda.append(videos_list)

    elif(detection == "apuntando_derecha"):
        apuntando_derecha.append(videos_list)

    elif(detection == "apuntando_izquierda"):
        apuntando_izquierda.append(videos_list)

    elif(detection == "caminando_derecha"):
        caminando_derecha.append(videos_list)

    elif(detection == "caminando_izquierda"):
        caminando_izquierda.append(videos_list)
    
#np.save("./"+DATA_PATH+"/golpe_derecha_mirada_derecha/golpe_derecha_mirada_derecha", np.array(golpe_derecha_mirada_derecha[0]))
#np.save("./"+DATA_PATH+"/golpe_izquierda_mirada_derecha/golpe_izquierda_mirada_derecha", np.array(golpe_izquierda_mirada_derecha[0]))
#np.save("./"+DATA_PATH+"/golpe_derecha_mirada_izquierda/golpe_derecha_mirada_izquierda", np.array(golpe_derecha_mirada_izquierda[0]))
#np.save("./"+DATA_PATH+"/golpe_izquierda_mirada_izquierda/golpe_izquierda_mirada_izquierda", np.array(golpe_izquierda_mirada_izquierda[0]))

print(len(caminando_derecha[0]))
#np.save("./"+DATA_PATH+"/apuntando_derecha/apuntando_derecha3", np.array(apuntando_derecha[0]))
#np.save("./"+DATA_PATH+"/apuntando_izquierda/apuntando_izquierda3", np.array(apuntando_izquierda[0]))

np.save("./"+DATA_PATH+"/caminando_derecha/caminando_derecha_t", np.array(caminando_derecha[0]))
#np.save("./"+DATA_PATH+"/caminando_izquierda/caminando_izquierda_t", np.array(caminando_izquierda[0]))
# del caminando derecha los ultimos 10
# del 5 los primeros 20


# np.save("./"+DATA_PATH+"/saludo/saludo", np.array(saludo[0]))
# np.save("./"+DATA_PATH+"/aplauso/aplauso", np.array(aplauso[0]))
# np.save("./"+DATA_PATH+"/juego_manos/juego_manos", np.array(juego_manos[0]))

cap.release()
cv2.destroyAllWindows()

