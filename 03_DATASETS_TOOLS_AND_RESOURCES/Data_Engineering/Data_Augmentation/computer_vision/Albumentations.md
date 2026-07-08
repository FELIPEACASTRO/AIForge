# Albumentations

## Description

Albumentations is a fast and flexible Python library for image data augmentation, specifically designed for Computer Vision and Deep Learning projects. It is widely used in industry and research due to its high speed and rich collection of transformations. Its main differentiator is performance optimization, being faster than most alternatives across many transformations.

## Statistics

Speed: Generally the fastest, with a median speedup of 2.64x compared to other libraries (such as imgaug, Kornia, torchvision) on image transformations. Throughput: Processes up to 10810 images/second (e.g., Brightness) on a single CPU thread. It is the fastest in 38 of 48 tested transformations.

## Features

Wide range of transformations (geometric, color, pixel). Support for different target types (bounding boxes, segmentation masks, keypoints). Simple API for composing complex pipelines (A.Compose and A.OneOf). Optimized for CPU using OpenCV, resulting in high speed.

## Use Cases

Training Computer Vision models for classification, object detection, and semantic segmentation. Machine Learning competitions (Kaggle) where data processing speed is crucial. Production applications where real-time data augmentation is required.

## Integration

Installation: `pip install -U albumentations`. Code example:\n```python\nimport albumentations as A\nimport cv2\n\ntransform = A.Compose([\n    A.RandomRotate90(),\n    A.Flip(),\n    A.OneOf([\n        A.MotionBlur(p=.2),\n        A.MedianBlur(blur_limit=3, p=0.1),\n    ], p=0.2),\n    A.HueSaturationValue(p=0.3),\n])\n\nimage = cv2.imread('image.jpg')\nimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)\naugmented_image = transform(image=image)['image']\n```

## URL

https://albumentations.ai/