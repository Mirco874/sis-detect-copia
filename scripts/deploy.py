import cv2
import mediapipe as mp
import time
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(34,132)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.load_weights("./detection_lstm_modelv0.01.h5")


actions = np.array([ 'asalto_cuchillo', 'asalto_pistola'])
sequence = []
sentence = []
predictions = []
threshold = 0.95

pose_detector_model = mp.solutions.pose
pose = pose_detector_model.Pose();

cap = cv2.VideoCapture(0);
prediction_time = 0;

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


def extract_keypoints(results):
    pose_landmarks = [];
    for result in results.pose_landmarks.landmark:
        land_mark = np.array([result.x, result.y, result.z, result.visibility]).flatten();
        pose_landmarks.append(land_mark);
        
    return np.array(pose_landmarks).flatten();


while True:
    # captura de la imagen
    sucesss, image = cap.read();
    # conversion al formato RGB
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB);
    #obtencion de la prediccion
    results = pose.process(imgRGB);
    if(results.pose_landmarks):
        # si existe alguna deteccion, agregala a la imagen BGR obtenida por la camara
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, pose_detector_model.POSE_CONNECTIONS)
        # de todas las marcas de una deteccion recuperar 


        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        sequence = sequence[-34:]

        if len(sequence) == 34:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)], res )
            predictions.append(np.argmax(res))

            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow("Image", image);
        key = cv2.waitKey(1)
        if key == 27:
            break

#[[132],[132],...,[132]]
#132 = 33 * 4 =   132
# the length of pose_landmark array will be 33, like results.pose_landmarks.landmark
# [[x,y,z,v],[x,y,z,v],....]


cap.release()
cv2.destroyAllWindows()