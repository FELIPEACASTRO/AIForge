# COCO (Common Objects in Context)

## Description
COCO (Common Objects in Context) is one of the largest and most influential computer vision datasets. It was designed to advance research in object detection, semantic and instance segmentation, recognition in context, and image captioning. The dataset is known for its detailed and complex annotations, including precise instance segmentation (pixel masks) for each object, as well as "stuff" annotations (formless things, such as grass or sky) and keypoints for people. The most widely used version for competitions is COCO 2017.

## Statistics
**Images:** 330K in total (>200K labeled). **Object Instances:** 1.5 million. **Categories:** 80 object categories and 91 "stuff" categories. **Annotations:** 5 captions per image and 250,000 people with keypoints. **2017 Version (most used):** Training Images (118K/18GB), Validation (5K/1GB), Test (41K/6GB), Unlabeled (123K/19GB). Training/Validation Annotations (241MB).

## Features
Large-scale object detection; Instance segmentation (pixel masks); "Stuff" segmentation (panoptic); Recognition in context; Keypoint detection for people; Image captions (5 per image).

## Use Cases
Training and evaluation of object detection models (for example, YOLO, Faster R-CNN); Image segmentation (semantic and instance); Image Captioning; Human Pose Detection (Keypoint Detection); Research in Computer Vision and Artificial Intelligence.

## Integration
The dataset can be downloaded directly through HTTP links to the ZIP files of the images and annotations (2014 and 2017 versions). The most recommended way for efficient download and manipulation of the annotations is through the **COCO API** (available on GitHub: https://github.com/cocodataset/cocoapi). The API provides tools to load, analyze, and visualize the annotations, as well as facilitating model evaluation. The use of tools such as `gsutil` is suggested to avoid downloading large ZIP files.

## URL
[https://cocodataset.org/](https://cocodataset.org/)
