import tensorflow as tf
import tensorflow_hub as hub
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard

model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']


lstm_model = Sequential()
lstm_model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(9,36)))
lstm_model.add(LSTM(64, return_sequences=False, activation='relu'))
#lstm_model.add(Dropout(0.1))  # Regularización con Dropout
lstm_model.add(Dense(64, activation='relu'))
lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dropout(0.1))  # Regularización con Dropout
lstm_model.add(Dense(4, activation='softmax'))

lstm_model.load_weights("./martes278.h5")

sequence = []
sentence = []
predictions = []
predicion_number = 1;
threshold = 0.70

actions = np.array(['apuntando_derecha', 'apuntando_izquierda','caminando_derecha','caminando_izquierda'])
# Dibujar los 17 landmarks en un segundo
#regresa los 13 puntos importantes
def loop_through_people(frame, keypoints_with_scores, confidence_threshold): 
    #un arreglo de 17 landmarks, cada uno con 2 coordenadas y 1 score
    for person in keypoints_with_scores:
        pose_data = np.array(person[5:])
        normalized_points=draw_keypoints(frame, pose_data, confidence_threshold)



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
while cap.isOpened():
    ret, frame = cap.read()
    
    # Resize image
    img = frame.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256,256)
    input_img = tf.cast(img, dtype=tf.int32)
    
    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    
    # Render keypoints 
    loop_through_people(frame, keypoints_with_scores, 0.5)
    
    person1 = keypoints_with_scores[0]
    pose_data = np.array(person1[5:])
    normalized_points=draw_keypoints(frame, pose_data, 0.5)

    sequence.append(np.array(normalized_points).flatten())
    #sequence = sequence[-10:]

    if len(sequence) == 9:
        res = lstm_model.predict(np.expand_dims(sequence, axis=0))[0]
        predictions.append(np.argmax(res))

        sequence = [];

        # if np.unique(predictions[-10:])[0]==np.argmax(res): 
            
        if res[np.argmax(res)] > threshold: 
            print(predicion_number, actions[np.argmax(res)], res[np.argmax(res)])
            predicion_number+=1;
            if len(sentence) > 0: 
                #if actions[np.argmax(res)] != sentence[-1]:
                    #print(actions[np.argmax(res)], res[np.argmax(res)])
                    sentence.append(actions[np.argmax(res)])
                    #sequence=[]
            else:
                sentence.append(actions[np.argmax(res)])

        # if len(sentence) > 5: 
        #     sentence = sentence[-5:]    



    cv2.imshow('Movenet Multipose', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()