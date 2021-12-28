from face_recognition.api import face_distance, face_encodings, face_locations
import numpy as np
import face_recognition as fr
import cv2
from engine import get_rostos

rostos_conhecidos, nome_dos_rostos = get_rostos()

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]

    localizacao_dos_rostos = fr.face_locations(rgb_frame)
    rosto_desconhecidos = fr.face_encodings(rgb_frame, localizacao_dos_rostos)

    for (top, right, bottom, left), rosto_desconhecidos in zip(localizacao_dos_rostos, rosto_desconhecidos):
        resultados = fr.compare_faces(rostos_conhecidos, rosto_desconhecidos)
        print(resultados)

        face_distances = fr.face_distance(rostos_conhecidos, rosto_desconhecidos)

        melhor_id = np.argmin(face_distances)
        if resultados[melhor_id]:
            nome = nome_dos_rostos[melhor_id]
        else:
            nome = "desconhecido"

        # Ao redor do rosto
        cv2.rectangle(frame ,(left, top), (right, bottom), (0, 0, 255), 2)

        # Embaixo do rosto
        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Texto
        cv2.putText(frame, nome, (left + 6, bottom -6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitkey(1) & 0xFF == ord('l'):
        break

video_capture.release()
cv2.destroyAllWindows()
    