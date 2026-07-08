# ChestX-ray14 Dataset (NIH Chest X-ray Dataset)

## Description

ChestX-ray14 is one of the largest and most widely used large-scale chest radiograph datasets. It was released by the National Institutes of Health (NIH) and contains 112,120 frontal X-ray images from 30,805 unique patients, annotated for the presence of up to 14 different thoracic pathologies. The annotations were generated automatically using text-mining techniques on radiology reports, which makes it a 'weakly labeled' dataset. It is widely used as a benchmark for developing deep learning models for the classification and localization of thoracic diseases.

## Statistics

It consists of 112,120 frontal chest X-ray images from 30,805 unique patients. The original images are in PNG format with a resolution of 1024 x 1024. The dataset totals approximately 45 GB.

## Features

The images are labeled for the presence of up to 14 thoracic pathologies: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, and Hernia. It also includes metadata such as age, sex, and view position (AP/PA).

## Use Cases

Multi-label classification of thoracic diseases, pneumonia detection, development of AI models for prioritizing urgent cases, fairness and bias studies in medical AI, and radiomic feature extraction (radiomics) for computer-aided diagnosis. Recent research (2023-2025) continues to use it for validating new *deep learning* models and *foundation models* in chest radiography.

## Integration

The dataset can be accessed and downloaded directly from the NIH (via Box.com) or through platforms such as Kaggle and Hugging Face, which provide pre-processed versions and simplified access tools. Integration generally involves the use of Python libraries such as PyTorch or TensorFlow, with custom data loaders to handle the multi-label structure and the metadata. Example of accessing the metadata file via Pandas (Kaggle): `import pandas as pd\ndf = pd.read_csv('/kaggle/input/nih-chest-xrays/Data_Entry_2017.csv')\nprint(df.head())`

## URL

https://nihcc.app.box.com/v/ChestXray-NIHCC