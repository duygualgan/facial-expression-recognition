from keras.models import load_model
import numpy as np
import cv2

# Eğitilmiş modeli yükle
model = load_model('emotion_detection_model_100epochs.h5')

# Tahmin yapmak istediğiniz resmi yükleyin ve boyutlandırın
image_path = 'denemeDuygu/duygu-deneme.jpg'  # Resmin dosya yolu
input_shape = (48, 48)  # Modelin giriş boyutu

# Resmi yükleyin ve boyut kontrolü yapın
image = cv2.imread(image_path)
if image is None:
    print("Resim yüklenemedi.")
    exit(1)

# Gri tonlamalı resme dönüştürün
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resmi boyutlandırın
resized_image = cv2.resize(gray_image, input_shape)

# Resmi modelin beklentilerine uygun hale getirin
input_data = np.expand_dims(resized_image, axis=0)
input_data = np.expand_dims(input_data, axis=-1)
input_data = input_data / 255.0  # Giriş verilerini normalize edin (0-1 aralığına getirin)

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# Tahmini yapın
predictions = model.predict(input_data)
emotion_label = np.argmax(predictions[0])
emotion = class_labels[emotion_label]

print("Tahmin edilen duygu: ", emotion)
