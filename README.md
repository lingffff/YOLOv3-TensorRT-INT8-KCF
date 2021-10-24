# YOLOv3-TensorRT-INT8-KCF

## Description
**YOLOv3-TensorRT-INT8-KCF** is a TensorRT Int8-Quantization implementation of YOLOv3 (and tiny) on NVIDIA Jetson Xavier NX Board. The dataset we provide is a red ball. So we also use this to drive a car to catch the red ball, along with KCF, a traditional Object Tracking method.
## Dependencies
### GPU server (e.g. GTX2080Ti)
- See [yolov3/requirements.txt](https://github.com/lingffff/YOLOv3-TensorRT-INT8-KCF/blob/master/yolov3/requirements.txt).
### NVIDIA Jetson Xavier NX
- TensorRT >= 7.0.0 (Pre-installed on NX.)
- OpenCV and opencv_contrib == 3.4.0 (See [here](https://blog.csdn.net/yuejing987/article/details/84986195) for help.)

## Tutorials
### 1. Train
On GPU server, not NX.
```bash
git clone https://github.com/lingffff/YOLOv3-TensorRT-INT8-KCF.git
cd YOLOv3-TensorRT-INT8-KCF
cd yolov3
# Download official pre-trained COCO darknet weights
sh weights/download_yolov3_weights.sh
```
Download redball dataset [here](https://disk.pku.edu.cn:443/link/D96BA7E9F3D894D225EFB2BE3DE74824), unzip and replace the folder **redball**. Then start training. Remove '--tiny' if you train YOLOv3 model.
```bash
python train.py --device 0 --tiny
```
### 2. Transfer
Now we get YOLOv3(tiny) weights in **weights/best.pt**. Transfer it to binary file **redball(-tiny).wts**, which convert weights to TensorRT for building inference engine.
```bash
python gen_wts.py --tiny
```
Then copy **./redball(-tiny).wts** to NX Board.
### 3. Build Engine
On NX Platform below.
```bash
git https://github.com/lingffff/YOLOv3-TensorRT-INT8-KCF.git
cd YOLOv3-TensorRT-INT8-KCF
# Put redball(-tiny).wts in YOLOv3-TensorRT-INT8-KCF
```
Build the project.
```bash
mkdir build
cd build
# YOLOv3: -DTINY=OFF, tiny: -DTINY=ON
cmake -DTINY=ON ..
make -j$(nproc)
```
Now we get executable file **build_engine** and **detect**.  
Run **build_engine**. Use **-s** argument to specify quantization options: int8, fp16, fp32(default).
```bash
./build_engine -s int8
```
### 4. Inference
Run **detect** to detect pictures or camera video. You can also check KCF tracking method here by other options below.  
```bash
./detect -p ../samples
```
Options:  
| Argument | Description |
| :-: | :-: |
| -p \<folder\> | Detect pictures in the folder. |
| -v  | Detect camera video stream.  |
| -t  | Detect video along with KCF tracking method.  |

## Benchmark
| Models | Device | BatchSize |	Mode | Input Size | Speed |
| :-: | :-: | :-: | :-: | :-: | :-:  |
| YOLOv3      | NX | 1 | FP32 | 416x416 | 85ms |
| YOLOv3      | NX | 1 | FP16 | 416x416 | 30ms |
| YOLOv3      | NX | 1 | INT8 | 416x416 | 26ms |
|             |    |   |      |         |      |
| YOLOv3-tiny | NX | 1 | FP32 | 416x416 | 26ms |
| YOLOv3-tiny | NX | 1 | FP16 | 416x416 | 19ms |
| YOLOv3-tiny | NX | 1 | INT8 | 416x416 | 20ms |
  
Wow! FP16 is amazing!!!

## TODO
- [ ] Convert weights to TensorRT by a more common way, like ONNX.
- [ ] Run detection and tracking multi-thread-ly.
- [ ] Implement a Quantization & Inference framework myself.

## Acknowledge

YOLOv3 Pytorch implementation from [ultralytics/yolov3](https://github.com/ultralytics/yolov3).  
YOLOv3 TensorRT implementation from [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx).  
TensorRT Int8 implementation from [NVIDIA/TensorRT/samples/sampleINT8](https://github.com/NVIDIA/TensorRT/tree/master/samples/sampleINT8).  

With my sincerely appreciation!

## About me  
Just call me **Al** (not ai but al. LOL.) / Albert / lingff.  
E-mail: ling@stu.pku.edu.cn  
Gitee: https://gitee.com/lingff  
CSDN: https://blog.csdn.net/weixin_43214408  