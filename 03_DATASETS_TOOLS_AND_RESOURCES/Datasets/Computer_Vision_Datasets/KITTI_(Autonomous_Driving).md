# KITTI (Autonomous Driving)

## Description
The **KITTI Vision Benchmark Suite** is one of the most influential and widely used datasets for research in **computer vision** and **autonomous driving**. Developed by the Karlsruhe Institute of Technology (KIT) and the Toyota Technological Institute at Chicago (TTIC), it provides multimodal sensor data from a moving vehicle. The collection vehicle was equipped with two high-resolution stereo cameras (color and grayscale), a **Velodyne 3D** laser scanner, and a **GPS/IMU** localization system, all synchronized at 10 Hz. The dataset was captured in urban, rural, and highway areas in the city of Karlsruhe, Germany, presenting complex and realistic scenarios. The main objective is to provide challenging benchmarks for real-time perception tasks.

## Statistics
**Training/Test Images:** The 3D Object Detection benchmark, for example, consists of 7,481 training images and 7,518 test images. **Annotated Objects:** The 3D Object Detection benchmark contains a total of 80,256 labeled objects. **Dataset Size:** The total size of the dataset (raw data) is substantial, with the Odometry dataset (Velodyne point clouds) alone reaching about **80 GB**. **Versions:** The original version is from 2012, with updates in 2015. The most recent and expanded version is **KITTI-360**, which offers greater scale and richer instance semantic annotations.

## Features
**Sensor Multimodality:** Combines stereo images (color and grayscale), 3D LiDAR point clouds, and localization/orientation data (GPS/IMU). **Realistic Scenarios:** Data collected under varied traffic and environmental conditions (city, rural, highway). **Diversity of Benchmarks:** Supports multiple benchmarks, including: Stereo, Optical Flow, Visual Odometry, 2D/3D Object Detection, Object Tracking, Semantic Segmentation, and Depth Prediction. **Detailed Annotations:** Includes high-quality annotations for 3D objects (cars, pedestrians, cyclists, etc.) and tracking labels.

## Use Cases
**Autonomous Driving:** Training and evaluation of perception systems for autonomous vehicles. **Object Detection and Tracking:** Development of algorithms to identify and track vehicles, pedestrians, and cyclists in 2D and 3D. **Visual Odometry and SLAM:** Evaluation of methods for motion estimation and simultaneous mapping using camera and LiDAR data. **Depth Prediction:** Training models to estimate the depth of a scene from stereo or monocular images. **Semantic Segmentation:** Classification of each image pixel or point cloud point into categories such as road, vegetation, vehicles, etc.

## Integration
Downloading the raw data and the specific benchmarks requires registration and login on the official website. The data is provided in file format (PNG for images, binary for point clouds, text for GPS/IMU, and XML for tracking labels). The official website provides a **download script** to facilitate obtaining the raw data. Development kits (devkits) in languages such as Python and MATLAB are provided to assist in reading, processing, and evaluating the data. The dataset is also available on platforms such as **TensorFlow Datasets** and **Hugging Face Datasets** for easier integration into machine learning pipelines.

## URL
[https://www.cvlibs.net/datasets/kitti/](https://www.cvlibs.net/datasets/kitti/)
