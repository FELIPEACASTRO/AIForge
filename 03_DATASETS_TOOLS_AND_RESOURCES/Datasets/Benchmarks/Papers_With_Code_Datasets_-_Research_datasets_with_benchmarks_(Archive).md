# Papers With Code Datasets - Research datasets with benchmarks (Archive)

## Description

Papers With Code (PWC) was a centralized, community-driven platform that catalogued Machine Learning (ML) research papers, their implementation code, associated datasets, and benchmark results tables. Its unique value proposition lay in directly connecting academic research (papers) with practical implementation (code), allowing researchers and practitioners to quickly find the State-of-the-Art (SOTA) for specific ML tasks. The platform was discontinued and its content was archived and partially migrated to Hugging Face, which is now the main access point for its data holdings. The platform's main focus was **Research Datasets with Benchmarks**, which allowed direct comparison of model performance on specific tasks.

## Statistics

- **Current Status**: Sunsetted in August 2025, with content archived and migrated to Hugging Face.
- **Data Volume (Historical)**: At its peak, the platform indexed more than **5,600** Machine Learning datasets [1], more than **950** unique ML tasks, more than **500** SOTA evaluation tables, and more than **8,500** papers with code [2].
- **Holdings on Hugging Face**: The public archive (`pwc-archive`) contains the final *snapshot* of the data, including more than **300,000** links between papers and code.
- **Adoption**: It was one of the main references for ML research, being cited in numerous academic papers for data collection and trend analysis [3].

## Features

- **Paper-Code-Dataset Connection**: A direct link between scientific papers, their code, and the datasets used, facilitating research reproducibility.
- **SOTA Benchmark Tables**: Organization of model results into leaderboards for various ML tasks, allowing quick identification of SOTA performance.
- **ML Task Taxonomy**: A hierarchical structure of tasks, sub-tasks, and ML areas (such as Computer Vision, Natural Language Processing, etc.) for navigation and discovery.
- **Open Data (Data Dumps)**: The complete PWC dataset was made publicly available on GitHub and archived on Hugging Face, allowing offline use and large-scale analysis.
- **Access API (Discontinued)**: A Python client existed for the read/write API, which is now maintained by the community for access to the archived data.

## Use Cases

- **Academic Research and Reproducibility**: Quickly find the code and data associated with a research paper, allowing the reproduction of results and validation of methodologies.
- **State-of-the-Art (SOTA) Analysis**: Identify the best-performing model for a specific Machine Learning task (e.g., image segmentation, machine translation) through the benchmark tables.
- **Dataset Discovery**: Explore and filter ML datasets by modality (vision, text, audio) or by task, a valuable resource for starting new projects.
- **AI Agent Development**: Use the data holdings (via Hugging Face or GitHub) to train and evaluate large language models (LLMs) and AI agents with knowledge of the ML task taxonomy and research results.

## Integration

Integration with the Papers With Code data holdings is done primarily through its public archive on GitHub or, in a more structured way, through the **Hugging Face Datasets Hub**, which hosts the `pwc-archive`.

**1. Accessing the Data Archive on Hugging Face (Recommended)**

Hugging Face hosts the PWC data archive, which can be accessed using the Python `datasets` library.

```python
from datasets import load_dataset

# Carrega o dataset de links entre artigos e código
dataset = load_dataset("pwc-archive/links-between-paper-and-code")

# Exemplo de acesso aos dados
print(dataset['train'][0])
# {'paper_id': '...', 'code_url': '...', 'repo_type': '...'}

# Outros datasets do arquivo podem ser explorados no Hugging Face Hub
# Ex: pwc-archive/all-papers-with-abstracts
```

**2. Direct Access to the Data Archive (GitHub)**

The `paperswithcode-data` repository on GitHub contains the complete *data dump*, which can be cloned and processed locally.

```bash
git clone https://github.com/paperswithcode/paperswithcode-data.git
```

The data is in JSON format and can be read with Python:

```python
import json

with open('paperswithcode-data/datasets.json', 'r') as f:
    datasets_data = json.load(f)

# Processar os dados conforme necessário
```

## URL

https://huggingface.co/pwc-archive
