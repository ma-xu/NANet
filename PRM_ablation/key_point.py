import torch
import cv2
# dog1.jpg
key_points = torch.Tensor(
    [
[[[[0.2857]],

         [[0.9643]]],


        [[[0.4464]],

         [[0.2500]]],


        [[[0.7500]],

         [[0.3571]]],


        [[[0.5179]],

         [[0.9821]]]],
[[[[1.2500e-01]],

         [[9.8214e-01]]],


        [[[3.7500e-01]],

         [[3.7500e-01]]],


        [[[1.2500e-01]],

         [[4.4643e-01]]],


        [[[1.7857e-07]],

         [[9.1071e-01]]]],
[[[[0.9464]],

         [[0.9643]]],


        [[[0.9286]],

         [[0.7500]]],


        [[[0.3214]],

         [[0.3750]]],


        [[[0.3750]],

         [[0.4107]]]],
[[[[0.4286]],

         [[0.6429]]],


        [[[0.1786]],

         [[0.3929]]],


        [[[0.4286]],

         [[0.3571]]],


        [[[0.1071]],

         [[0.3571]]]],
[[[[0.2500]],

         [[0.4643]]],


        [[[0.2500]],

         [[0.3214]]],


        [[[0.3571]],

         [[0.3571]]],


        [[[0.9643]],

         [[0.9643]]]],
[[[[0.3214]],

         [[0.3214]]],


        [[[0.6071]],

         [[0.8571]]],


        [[[0.8214]],

         [[0.6071]]],


        [[[0.7500]],

         [[0.6786]]]],
[[[[0.1429]],

         [[0.3571]]],


        [[[0.5714]],

         [[0.5714]]],


        [[[0.3214]],

         [[0.3571]]],


        [[[0.1429]],

         [[0.4286]]]],
[[[[4.2857e-01]],

         [[3.5714e-01]]],


        [[[7.1429e-01]],

         [[5.0000e-01]]],


        [[[9.2857e-01]],

         [[7.1429e-07]]],


        [[[2.8572e-01]],

         [[3.5714e-01]]]],
[[[[0.2857]],

         [[0.3571]]],


        [[[0.2857]],

         [[0.2857]]],


        [[[0.1429]],

         [[0.4286]]],


        [[[0.2857]],

         [[0.3571]]]],
[[[[5.0000e-01]],

         [[2.8572e-01]]],


        [[[9.2857e-01]],

         [[9.2857e-01]]],


        [[[1.4286e-01]],

         [[3.5714e-01]]],


        [[[3.5714e-01]],

         [[7.1429e-07]]]],
[[[[0.2143]],

         [[0.3571]]],


        [[[0.2857]],

         [[0.2143]]],


        [[[0.6429]],

         [[0.4286]]],


        [[[0.9286]],

         [[0.0714]]]],
[[[[7.1429e-02]],

         [[9.2857e-01]]],


        [[[2.8572e-01]],

         [[3.5714e-01]]],


        [[[7.1429e-07]],

         [[4.2857e-01]]],


        [[[4.2857e-01]],

         [[5.0000e-01]]]],
[[[[0.2143]],

         [[0.3571]]],


        [[[0.2143]],

         [[0.3571]]],


        [[[0.2857]],

         [[0.2857]]],


        [[[0.3571]],

         [[0.3571]]]],
[[[[1.4286e-06]],

         [[1.4286e-06]]],


        [[[2.8572e-01]],

         [[4.2857e-01]]],


        [[[2.8572e-01]],

         [[4.2857e-01]]],


        [[[1.4286e-06]],

         [[1.4286e-06]]]],
[[[[0.1429]],

         [[0.4286]]],


        [[[0.2857]],

         [[0.2857]]],


        [[[0.2857]],

         [[0.4286]]],


        [[[0.4286]],

         [[0.4286]]]],
[[[[4.2857e-01]],

         [[4.2857e-01]]],


        [[[8.5714e-01]],

         [[1.4286e-06]]],


        [[[8.5714e-01]],

         [[1.4286e-06]]],


        [[[8.5714e-01]],

         [[1.4286e-06]]]]]
)


key_points = torch.Tensor([

])

"""
# bird.jpeg
key_points = torch.Tensor([
[[[[0.9643]],

         [[0.5893]]],


        [[[0.4464]],

         [[0.6429]]],


        [[[0.5536]],

         [[0.3036]]],


        [[[0.6964]],

         [[0.9821]]]],
[[[[0.7857]],

         [[0.8214]]],


        [[[0.1964]],

         [[0.8214]]],


        [[[0.1964]],

         [[0.8214]]],


        [[[0.8214]],

         [[0.8214]]]],
[[[[0.8929]],

         [[0.6250]]],


        [[[0.8036]],

         [[0.6429]]],


        [[[0.3393]],

         [[0.4643]]],


        [[[0.9821]],

         [[0.6607]]]],
[[[[0.1786]],

         [[0.6429]]],


        [[[0.2143]],

         [[0.8571]]],


        [[[0.2143]],

         [[0.6429]]],


        [[[0.9286]],

         [[0.7143]]]],
[[[[0.3929]],

         [[0.8571]]],


        [[[0.3571]],

         [[0.7143]]],


        [[[0.6429]],

         [[0.5000]]],


        [[[0.9643]],

         [[0.5714]]]],
[[[[0.4286]],

         [[0.8214]]],


        [[[0.1429]],

         [[0.7857]]],


        [[[0.0714]],

         [[0.7857]]],


        [[[0.7143]],

         [[0.2857]]]],
[[[[8.5714e-01]],

         [[6.4286e-01]]],


        [[[3.5714e-07]],

         [[9.6429e-01]]],


        [[[3.5715e-02]],

         [[7.1429e-01]]],


        [[[6.7857e-01]],

         [[7.1429e-02]]]],
[[[[9.2857e-01]],

         [[6.4286e-01]]],


        [[[2.1429e-01]],

         [[7.1429e-01]]],


        [[[6.4286e-01]],

         [[7.1429e-07]]],


        [[[2.1429e-01]],

         [[7.1429e-01]]]],
[[[[0.2143]],

         [[0.8571]]],


        [[[0.2143]],

         [[0.8571]]],


        [[[0.1429]],

         [[0.7857]]],


        [[[0.1429]],

         [[0.7143]]]],
[[[[0.5714]],

         [[0.9286]]],


        [[[0.9286]],

         [[0.2857]]],


        [[[0.7857]],

         [[0.9286]]],


        [[[0.6429]],

         [[0.9286]]]],
[[[[0.5000]],

         [[0.5714]]],


        [[[0.2143]],

         [[0.9286]]],


        [[[0.9286]],

         [[0.3571]]],


        [[[0.9286]],

         [[0.9286]]]],
[[[[0.0714]],

         [[0.7857]]],


        [[[0.2143]],

         [[0.6429]]],


        [[[0.2857]],

         [[0.6429]]],


        [[[0.2857]],

         [[0.7143]]]],
[[[[0.2857]],

         [[0.8571]]],


        [[[0.2143]],

         [[0.7143]]],


        [[[0.3571]],

         [[0.8571]]],


        [[[0.3571]],

         [[0.7857]]]],
[[[[1.4286e-06]],

         [[1.4286e-01]]],


        [[[2.8572e-01]],

         [[7.1429e-01]]],


        [[[7.1429e-01]],

         [[8.5714e-01]]],


        [[[1.4286e-06]],

         [[1.4286e-01]]]],
[[[[0.2857]],

         [[0.7143]]],


        [[[0.2857]],

         [[0.7143]]],


        [[[0.2857]],

         [[0.7143]]],


        [[[0.7143]],

         [[0.4286]]]],
[[[[2.8572e-01]],

         [[8.5714e-01]]],


        [[[2.8572e-01]],

         [[1.4286e-06]]],


        [[[2.8572e-01]],

         [[1.4286e-06]]],


        [[[2.8572e-01]],

         [[1.4286e-06]]]]
])
"""

# 0:hight 1:weight
key_points = key_points.view((16,4,2))
print(key_points.shape)
image_path = "GAMCAM/bird.jpeg"
im = cv2.imread(image_path)
h,w,c  = im.shape
print(im.shape)
for i in range(16):
    im = cv2.imread(image_path)
    keys = key_points[i,:,:] # shape [4,2]
    for j in range(4):
        H,W  = keys[j,:]
        cv2.circle(im, (int(H*h), int(W*w)), 3, (0, 0, 255), 3)
        print(int(H*h), int(W*w))
    print("______________________")

    write_path = "/Users/melody/Downloads/points/bird"+str(i)+".jpg"
    cv2.imwrite(write_path,im)

    # cv2.namedWindow("image")
    # cv2.imshow('image', im)
    # cv2.waitKey(2000)  # show 10000 ms/10s and then disappear
    # cv2.destroyAllWindows()
