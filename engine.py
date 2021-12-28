import face_recognition as fr

def reconhece_face(url_foto):
        foto = fr.load_image_file(url_foto)
        rostos = fr.face_encodings(foto)
        if(len(rostos) > 0):
                return True, rostos
            
        return False, []

def get_rostos():
    rostos_conhecidos = []
    nomes_dos_rostos = []
    eu1 = reconhece_face("./img/eu1.jpg")
    if(eu1[0]):
        rostos_conhecidos.append(eu1[1][0])
        nomes_dos_rostos.append("Feng")

    eu2 = reconhece_face("./img/eu2.jpg")
    if(eu2[0]):
        rostos_conhecidos.append(eu2[1][0])
        nomes_dos_rostos.append("Feng_2")

    return rostos_conhecidos, nomes_dos_rostos