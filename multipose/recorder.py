import cv2
import os
from datetime import datetime

# Crear la carpeta "grabaciones" si no existe
if not os.path.exists("grabaciones"):
    os.makedirs("grabaciones")

# Inicializar la captura de video
video_capture = cv2.VideoCapture(0)

# Obtener la resolución predeterminada del dispositivo de captura
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))

# Crear el objeto VideoWriter para guardar el video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output = None
is_recording = False
is_paused = False

while True:
    # Leer el siguiente frame del video
    ret, frame = video_capture.read()

    # Mostrar el frame en una ventana
    cv2.imshow("Video", frame)

    # Si se presiona la tecla 'r', comenzar o detener la grabación
    key = cv2.waitKey(1)
    if key == ord('r'):
        if not is_recording:
            # Obtener la fecha y hora actual
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

            # Crear el nombre del archivo usando la fecha y hora actual
            filename = f"grabaciones/recording_{timestamp}.avi"

            # Crear el objeto VideoWriter con el nuevo nombre de archivo
            output = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

            is_recording = True
            print("Iniciando grabación")
        else:
            is_recording = False
            output.release()
            print("Deteniendo grabación")

    # Si se presiona la tecla 'p', pausar o reanudar la grabación
    if key == ord('p'):
        if is_recording:
            if not is_paused:
                is_paused = True
                print("Grabación pausada")
            else:
                is_paused = False
                print("Grabación reanudada")

    # Si se está grabando y no está en pausa, escribir el frame en el archivo de salida
    if is_recording and not is_paused:
        output.write(frame)

    # Si se presiona la tecla 'q', salir del bucle
    if key == ord('q'):
        break

# Liberar los recursos
video_capture.release()
cv2.destroyAllWindows()
