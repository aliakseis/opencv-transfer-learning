# opencv-transfer-learning

Taken from https://answers.opencv.org/answers/191837/revisions/

How to Install OpenCV on Ubuntu 18.04: https://linuxize.com/post/how-to-install-opencv-on-ubuntu-18-04/

Vcpkg can be used on Windows:

```
vcpkg install --recurse opencv[contrib,dnn,ipp,ffmpeg,tbb,sfm,vtk,nonfree]:x64-windows
```

SqueezeNet stuff:
```
wget https://raw.githubusercontent.com/DeepScale/SqueezeNet/b5c3f1a23713c8b3fd7b801d229f6b04c64374a5/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel
wget https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/squeezenet_v1.1.prototxt
```
