# E-puck Autonomous Block Search & CIFAR-10 CNN Object Classification

**Module:** Applied Robotics and AI Project  
**Institution:** University of Hertfordshire  
**Author:** LAW LI KANG RAIN  

## 1. Project Overview
This repository contains the Webots simulation world, Python robot controller, and TensorFlow/Keras Convolutional Neural Network (CNN) implementation for **Project Part 1** and **Project Part 2**:
- **Part 1 (Webots Autonomous Block Search):** Autonomous exploration using an e-puck robot to detect red, blue, and green blocks (`0.3 m x 0.3 m x 0.1 m`), print first-encounter responses (`"I see red/blue/green"`), log encounter summaries, and record multi-round search durations.
- **Part 2 (CIFAR-10 CNN & Edge Optimization):** External CNN training on the CIFAR-10 dataset (`32x32x3`), custom image inference, e-puck onboard camera capture classification, and algorithm optimization (camera pre-filtering + TFLite post-training quantization).

## 2. Repository Structure
- `webots_simulation/worlds/`: Custom Webots R2023b arena (`1.5 m x 1.5 m`) with randomly positioned target blocks (`z = 0.05 m`, `roughness = 1.0`, `metalness = 0.0`, active `boundingObject`).
- `webots_simulation/controllers/e-puck_search/`: Modular Python controller (`e-puck_search.py`) using 8 IR proximity sensors (`ps0`–`ps7`), forward CMOS RGB camera, and differential-drive motors.
- `cnn_model/`: Google Colab notebook (`CIFAR10_CNN_Part2.ipynb`), trained Keras/TFLite models, and test images (`sample_image.png`, custom images, and Webots camera capture).

## 3. Hardware & Simulation Parameters
- **Simulator:** Webots R2023b (`TIME_STEP = 64 ms`)
- **Robot Platform:** GCtronic / EPFL e-puck (`MAX_SPEED = 6.28 rad/s`)
- **Deep Learning Environment:** Google Colab (T4 GPU), Python 3, TensorFlow / Keras
