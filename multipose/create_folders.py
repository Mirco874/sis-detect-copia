import os;
import numpy as np;

DATA_PATH = os.path.join('TRAIN_DATA'); 
# detections = np.array(['caminando_izquierda', 'caminando_derecha', 'pelea_izquierda', 'pelea_derecha'])
detections = np.array(['golpe_derecha_mirada_derecha', 'golpe_izquierda_mirada_derecha','golpe_derecha_mirada_izquierda', 'golpe_izquierda_mirada_izquierda'])
for detection in detections: 
    os.makedirs(os.path.join(DATA_PATH, detection))


