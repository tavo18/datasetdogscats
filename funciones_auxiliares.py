# coding=utf-8
import cv2
import os
import numpy as np
from tqdm import tqdm

def read_image(file_path, filas, columnas):
 img = cv2.imread(file_path, cv2.IMREAD_COLOR)
 return cv2.resize(img, (filas, columnas), interpolation=cv2.INTER_CUBIC)

def preprocesar(images, filas, columnas, canales):
  m = len(images)
  X = np.zeros((filas*columnas*canales,m),dtype=np.uint8)
  y = np.zeros((1, m))
  for i, image_file in tqdm(enumerate(images),total=m, unit="imagenes"):
    img = read_image(image_file, filas, columnas) # Lectura de la imagen
    img = img/255 # Estandarización
    img = img.reshape(filas*columnas*canales) # "Aplanamiento" y almacenamiento
    X[:,i] = img
    if 'dog' in image_file.lower():
      y[0, i] = 1
    elif 'cat' in image_file.lower():
      y[0, i] = 0
  return X, y