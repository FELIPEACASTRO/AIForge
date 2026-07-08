# Flickr30k

## Description
Flickr30k is a widely used benchmark dataset for sentence-based image description (Image Captioning) and text-image retrieval (Text-Image Retrieval) tasks. It consists of 31,783 color images obtained from Flickr, each paired with five distinct captions written by humans. An extended version, **Flickr30k Entities**, increases the usefulness of the dataset by adding bounding box annotations and coreference chains for the entities mentioned in the captions, enabling alignment between image regions and phrases in the descriptions. It is one of the most important datasets for the development of multimodal models.

## Statistics
*   **Images:** 31,783 color images.
*   **Captions:** 158,915 English captions (5 per image).
*   **Extended Version (Flickr30k Entities):** Includes 244,000 coreference chains and 276,000 bounding boxes.
*   **Recent Versions (2024):** The dataset continues to be used as a basis for new extensions, such as **Flickr30K-CFQ** (2024) and **FM30K** (2024, with captions in Brazilian Portuguese), indicating its continued relevance.

## Features
*   **Multimodality:** Combines visual data (images) and textual data (captions).
*   **Diversity:** Images capture people engaged in everyday activities, ensuring a wide variety of scenes and concepts.
*   **Multiple Captions:** Each image has 5 independent captions, which allows for a more robust evaluation of *image captioning* models.
*   **Entities Extension:** The Flickr30k Entities version adds 244 thousand coreference chains and 276 thousand manually annotated bounding boxes, linking textual entities to visual regions.

## Use Cases
*   **Image Captioning:** Automatic generation of textual descriptions for images.
*   **Text-Image Retrieval:** Searching for images from a textual description and vice versa.
*   **Visual Entity Grounding:** Alignment of phrases in captions with specific regions of the image (especially with the Flickr30k Entities version).
*   **Multimodal Models:** Training and evaluation of models that integrate computer vision and natural language processing (CV+NLP).
*   **Multimodal Translation:** Used as a basis for multilingual datasets, such as Multi30K and FM30K (Brazilian Portuguese).

## Integration
The original dataset can be obtained through the official University of Illinois page, which provides links to the images and captions. For the publicly distributable version (image links + captions), the process involves downloading annotation files and obtaining the images directly from Flickr (subject to Flickr's Terms of Use).

**Integration via Hugging Face (Recommended):**
The most modern and simplest way to integrate the dataset is through the Hugging Face `datasets` library, which already manages the annotations and data structure:
```python
from datasets import load_dataset

# For the base version
dataset = load_dataset("nlphuji/flickr30k")

# For the version with Brazilian Portuguese captions (FM30K)
# dataset = load_dataset("FrameNetBrasil/FM30K")
```
**Note:** The user must ensure they have the Flickr images for non-commercial use, since the annotation files generally contain only the links or IDs of the images. Platforms such as Kaggle often provide pre-packaged versions of the dataset, but the official source should be consulted for the terms of use.

## URL
[https://shannon.cs.illinois.edu/DenotationGraph/data/index.html](https://shannon.cs.illinois.edu/DenotationGraph/data/index.html)
