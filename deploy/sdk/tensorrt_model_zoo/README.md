# test tensorrt
## 模型转换
```
trtexec --onnx=yolo.onnx --saveEngine=yolo.trt --fp16
```
## build
```
mkdir build && cd build
cmake ..
make
```