import torch
import cv2


image_path = "GAMCAM/dog2.jpg"
im = cv2.imread(image_path)
im = cv2.resize(im, (224,224))
cv2.imwrite(image_path,im)
