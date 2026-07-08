# Adult (Census Income) Dataset

## Description
The Adult dataset, also known as the Census Income Dataset, is a classic dataset extracted from the 1994 U.S. Census database. The main goal is to predict whether an individual's annual income exceeds $50,000 per year based on 14 demographic and employment attributes. It is widely used in classification tasks and in Fair Machine Learning research due to its inclusion of sensitive attributes such as race and sex. Although the task name was "Employment Data", research identified that the most relevant and popular dataset in this context, especially in Machine Learning, is the "Adult (Census Income)".

## Statistics
The dataset contains 48,842 instances (samples) and 14 attributes (features). The compressed data file has a size of 605.7 KB. The original version was extracted from the 1994 Census.

## Features
The dataset has 14 attributes, including: age (continuous), work class (categorical), education (categorical and numeric), marital status (categorical), occupation (categorical), relationship (categorical), race (categorical), sex (binary), capital gain (continuous), capital loss (continuous), hours per week (continuous), and native country (categorical). The target variable is binary: whether income exceeds $50K/year or not. Contains missing values.

## Use Cases
- **Binary Classification:** Income prediction (>50K or <=50K).
- **Fair Machine Learning (Fair ML):** Evaluation of bias and discrimination in AI models, using sensitive attributes such as race and sex.
- **Data Analysis:** Study of demographic and employment factors that influence income.
- **Algorithm Testing:** Benchmark for new classification algorithms.

## Integration
The dataset can be downloaded directly from the UCI Machine Learning repository. For Python users, the most recommended integration is through the `ucimlrepo` package.
1.  **Installation:** `pip install ucimlrepo`
2.  **Usage in Python:**
    ```python
    from ucimlrepo import fetch_ucirepo 
    
    # Fetch the Adult dataset (ID=2)
    adult = fetch_ucirepo(id=2) 
    
    # Data (as pandas dataframes)
    X = adult.data.features 
    y = adult.data.targets 
    
    # Metadata and variable information are also available
    # print(adult.metadata) 
    # print(adult.variables)
    ```

## URL
[https://archive.ics.uci.edu/dataset/2/adult](https://archive.ics.uci.edu/dataset/2/adult)
