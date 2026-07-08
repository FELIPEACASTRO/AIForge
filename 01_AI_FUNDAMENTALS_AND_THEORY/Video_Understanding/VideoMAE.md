# VideoMAE

## Description

VideoMAE (Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training) is a self-supervised learning framework that extends the concept of Masked Autoencoders (MAE) to the video domain. Its unique value proposition is to demonstrate that MAEs are data-efficient learners for video pre-training, using an extremely high masking ratio (90%-95%) and a tube masking strategy to force the model to learn spatiotemporal coherence, overcoming the redundancy inherent in videos.

## Statistics

The base model (VideoMAE-Base) reaches about **80.9%** Top-1 accuracy and **94.7%** Top-5 accuracy on the Kinetics-400 test set (after fine-tuning). More recent versions (VideoMAE V2 Huge) achieve up to **86.6%** Top-1 accuracy and **97.1%** Top-5 accuracy on Kinetics-400.

## Features

Tube Masking Strategy; High Masking Ratio (90%-95%); Transformer (ViT) architecture with encoder and lightweight decoder; Data-efficient self-supervised learning.

## Use Cases

Video Action Classification (Kinetics-400); Action Detection; Human Activity Recognition (HAR) in assistive robotics and surveillance; Backbone for various video understanding tasks.

## Integration

The model is easily accessible and usable through the Hugging Face Transformers library.\n\n```python\nfrom transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification\nimport torch\n\n# Load the feature extractor and the pre-trained model\nfeature_extractor = VideoMAEFeatureExtractor.from_pretrained(\"MCG-NJU/videomae-base-finetuned-kinetics\")\nmodel = VideoMAEForVideoClassification.from_pretrained(\"MCG-NJU/videomae-base-finetuned-kinetics\")\n\n# Usage example (replace 'video_data' with the frames of your video)\n# inputs = feature_extractor(video_data, return_tensors=\"pt\")\n# with torch.no_grad():\n#     outputs = model(**inputs)\n# logits = outputs.logits\n# predicted_class_idx = logits.argmax(-1).item()\n# print(f\"Predicted Class: {model.config.id2label[predicted_class_idx]}\")\n```

## URL

https://github.com/MCG-NJU/VideoMAE