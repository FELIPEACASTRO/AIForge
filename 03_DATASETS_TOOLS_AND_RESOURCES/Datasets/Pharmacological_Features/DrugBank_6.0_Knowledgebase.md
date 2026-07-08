# DrugBank 6.0 Knowledgebase

## Description

DrugBank 6.0 is a comprehensive, open-access knowledgebase that combines detailed information about drugs and drug targets (such as proteins and genes). Version 6.0, released in 2024, significantly expanded the number of FDA-approved drugs and drug-drug interaction (DDI) entries, making it a fundamental resource for research in pharmacology and medical informatics. It includes data on the chemical, pharmacological, pharmacokinetic, and pharmacodynamic properties of drugs.

## Statistics

A total of 15,468 small-molecule drugs, 4,308 biotech drugs, and 4,772 approved drugs. Version 6.0 added 2,550 new enzyme-drug entries, 1,560 transporter-drug entries, and 550 carrier-drug entries. More than 30 million views per year.

## Features

Chemical and structural properties (SMILES, InChI), pharmacological data (mechanism of action, metabolism), drug-drug interactions (DDIs), drug targets (enzymes, transporters, carriers), and clinical information. Version 6.0 increased the number of FDA-approved drugs by 72% and DDI data by 300%.

## Use Cases

Prediction of drug-drug interactions (DDIs), identification of new drug targets, drug repurposing, ADMET property modeling (Absorption, Distribution, Metabolism, Excretion, and Toxicity), and development of clinical decision support systems.

## Integration

Access via web interface (DrugBank Online) and API for programmatic integration. The complete database can be downloaded for research and software development purposes under a license. The Clinical API offers a robust drug-drug interaction checker and advanced search features. Example of API access (conceptual):
```python
import requests

api_key = "SUA_CHAVE_API"
drug_name = "Aspirin"
url = f"https://api.drugbank.com/v1/drugs/{drug_name}"
headers = {"Authorization": f"Bearer {api_key}"}

response = requests.get(url, headers=headers)
if response.status_code == 200:
    data = response.json()
    print(f"Mecanismo de Ação: {data['mechanism_of_action']}")
```

## URL

https://go.drugbank.com/