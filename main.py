import cv2
import numpy as np
from keras.models import load_model

video = cv2.VideoCapture('video/video_teste.mp4')

model = load_model('model/keras_model.h5', compile=False)

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

labels = ["headset", "Desconhecido", "livro", "caneta"]


while True:
    _, img = video.read()
    image_resized = cv2.resize(img, (224, 224))
    image_array = np.asarray(image_resized)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction_object = model.predict(data)
    indexValue = np.argmax(prediction_object)

    cv2.putText(img, str(labels[indexValue]),(50,50), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
    cv2.imshow("Imagem", img)

    if cv2.waitKey(1) & 0xFF == 27:
        cv2.destroyAllWindows()
        exit()

