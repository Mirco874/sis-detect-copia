import cv2
import os
from datetime import datetime
import time

# Crear la carpeta "grabaciones" si no existe
if not os.path.exists("grabaciones_automaticas"):
    os.makedirs("grabaciones_automaticas")

# Inicializar la captura de video
video_capture = cv2.VideoCapture(0)

# Obtener la resolución predeterminada del dispositivo de captura
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

# Crear el objeto VideoWriter para guardar el video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output = None
is_recording = False
start_time = time.time()
total_duration = 10  # Duración total de grabación en segundos
pause_interval = 2000  # Intervalo de pausa en segundos
pause_duration = 4000  # Duración de la pausa en segundos
previous_time = time.time();
elapsed_time = time.time();

while True:
    # Leer el siguiente frame del video
    ret, frame = video_capture.read()

    # Mostrar el frame en una ventana
    cv2.imshow("Video", frame)

    # Si se presiona la tecla 'r', comenzar o detener la grabación
    key = cv2.waitKey(1)

    if key == ord('r'):        
        if is_recording:
            is_recording = False
            output.release()
            print("Deteniendo grabación")
        else:
            # Obtener la fecha y hora actual
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

            # Crear el nombre del archivo usando la fecha y hora actual
            filename = f"grabaciones_automaticas/recording_{timestamp}.avi"

            # Crear el objeto VideoWriter con el nuevo nombre de archivo
            output = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

            is_recording = True
            start_time = time.time()
            print("Iniciando grabación")

    # Si se está grabando, escribir el frame en el archivo de salida
    if is_recording:
        output.write(frame)
        current_time = time.time()

        if(current_time - previous_time >= 10):            
            print("pausando")
            cv2.waitKey(pause_duration)    
            previous_time = current_time;
            print("reanudado")
        

    # Si se presiona la tecla 'q', salir del bucle
    if key == ord('q'):
        break

# Liberar los recursos
video_capture.release()
cv2.destroyAllWindows()
