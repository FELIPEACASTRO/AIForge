# DrugBank 6.0 Knowledgebase

## Description

O DrugBank 6.0 é uma base de conhecimento abrangente e de acesso aberto que combina informações detalhadas sobre medicamentos e alvos medicamentosos (como proteínas e genes). A versão 6.0, lançada em 2024, expandiu significativamente o número de medicamentos aprovados pela FDA e as entradas de interações medicamentosas (DDIs), tornando-se um recurso fundamental para a pesquisa em farmacologia e informática médica. Inclui dados sobre propriedades químicas, farmacológicas, farmacocinéticas e farmacodinâmicas dos medicamentos.

## Statistics

Total de 15.468 medicamentos de pequenas moléculas, 4.308 medicamentos biotecnológicos e 4.772 medicamentos aprovados. A versão 6.0 adicionou 2.550 novas entradas de enzima-medicamento, 1.560 de transportador-medicamento e 550 de carreador-medicamento. Mais de 30 milhões de visualizações/ano.

## Features

Propriedades químicas e estruturais (SMILES, InChI), dados farmacológicos (mecanismo de ação, metabolismo), interações medicamentosas (DDIs), alvos medicamentosos (enzimas, transportadores, carreadores) e informações clínicas. A versão 6.0 aumentou em 72% o número de medicamentos aprovados pela FDA e em 300% os dados de DDI.

## Use Cases

Previsão de interações medicamentosas (DDIs), identificação de novos alvos medicamentosos, reposicionamento de medicamentos, modelagem de propriedades ADMET (Absorção, Distribuição, Metabolismo, Excreção e Toxicidade) e desenvolvimento de sistemas de suporte à decisão clínica.

## Integration

Acesso via interface web (DrugBank Online) e API para integração programática. O banco de dados completo pode ser baixado para fins de pesquisa e desenvolvimento de software, mediante licença. A API Clinical oferece um verificador robusto de interações medicamentosas e recursos avançados de pesquisa. Exemplo de acesso via API (conceitual):
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