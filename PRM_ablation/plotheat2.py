import torch
import cv2
import sys
import numpy as np

img = cv2.imread("GAMCAM/bird2.png")

img = np.float32(cv2.resize(img,(224,224)))/255
data_num="1"
data = np.loadtxt("distance/"+data_num+".txt")
print(np.min(data))
print(np.max(data))
data = (data-np.min(data))/(np.max(data)-np.min(data))
w,h = data.shape
scale = 224/w

data = torch.from_numpy(data)
data = data.unsqueeze(dim=0).unsqueeze(dim=0)
data = torch.nn.functional.upsample(data,scale_factor=scale,mode="nearest")
data = data.squeeze(dim=0).squeeze(dim=0)
data = data.numpy()
print(data.shape)



heatmap = cv2.applyColorMap(np.uint8(255*data),cv2.COLORMAP_JET)
heatmap = np.float32(heatmap)/255

# heatmap = heatmap.repeat(scale, axis=0).repeat(scale, axis=1)
cam =heatmap+np.float32(img)
cam = cam/np.max(cam)
cv2.imwrite("out"+data_num+".jpg",np.uint8(255*cam))


