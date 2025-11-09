# Something-Something V2 (Temporal Reasoning)

## Description
O dataset **Something-Something (versão 2)** é uma coleção de **220.847 clipes de vídeo** rotulados de humanos realizando ações básicas e predefinidas com objetos do cotidiano. Ele foi projetado especificamente para treinar modelos de aprendizado de máquina em uma compreensão detalhada de gestos humanos e interações, exigindo **raciocínio temporal** para distinguir entre ações semelhantes, como "levantar algo completamente" versus "levantar algo e depois soltar". O dataset é crucial para o avanço da pesquisa em **reconhecimento de ação em vídeo** e **compreensão de bom senso visual**.

## Statistics
- **Tamanho Total:** 19.4 GB (vídeos).
- **Amostras (Vídeos):** 220.847 clipes de vídeo.
- **Categorias de Ação:** 174 classes.
- **Atores de Crowdsourcing:** Mais de 1.300 únicos.
- **Versão:** V2 (Versão 2), lançada em 2018, sendo a mais utilizada para benchmarks.
- **Resolução:** 240px de altura.

## Features
- **Foco em Raciocínio Temporal:** As ações são definidas por frases-modelo (e.g., "Putting [something] onto [something]"), onde a ordem e a relação temporal entre os objetos e as ações são cruciais.
- **Grande Escala:** Contém 220.847 vídeos curtos e aparados.
- **Diversidade de Ações:** Abrange 174 categorias de ações distintas.
- **Anotações Detalhadas:** Inclui anotações de objetos (318.572 anotações, 30.408 objetos únicos) para os conjuntos de treinamento e validação.
- **Alta Qualidade:** Cada vídeo foi verificado por cinco atores de crowdsourcing diferentes para garantir a precisão do rótulo.
- **Formato:** Vídeos em formato WebM (codec VP9) com altura de 240px.

## Use Cases
- **Reconhecimento de Ação em Vídeo:** É o principal benchmark para modelos que buscam classificar ações em vídeos, especialmente aquelas que dependem de contexto e ordem temporal.
- **Raciocínio Temporal:** Treinamento e avaliação de modelos de redes neurais (como a Temporal Relation Network - TRN) capazes de aprender e raciocinar sobre relações temporais entre quadros de vídeo.
- **Visão Computacional e Robótica:** Desenvolvimento de sistemas que exigem uma compreensão fina de manipulações de objetos e gestos humanos para tarefas de imitação ou interação.
- **Modelos de Linguagem de Vídeo (Video LLMs):** Utilizado em pesquisas recentes (2024-2025) para aprimorar a capacidade de raciocínio temporal e compreensão de vídeos longos em modelos multimodais.

## Integration
O dataset pode ser acessado através do site oficial da Qualcomm (TwentyBN), onde os vídeos são fornecidos em um grande arquivo TGZ, dividido em partes de 1 GB (tamanho total de 19.4 GB). As anotações (rótulos e divisões) são fornecidas em arquivos JSON separados.

**Instruções de Uso (Exemplo com Hugging Face):**
Para uso em pesquisa e desenvolvimento, a versão V2 está disponível no Hugging Face Datasets, facilitando a integração com bibliotecas de aprendizado de máquina:

```python
from datasets import load_dataset

# Carrega o dataset (apenas metadados, os vídeos precisam ser baixados separadamente)
dataset = load_dataset("HuggingFaceM4/something_something_v2")

# Para preparar o dataset para modelos de reconhecimento de ação,
# é comum seguir as instruções de bibliotecas como o MMAction2.
# O download dos vídeos deve ser feito a partir da fonte primária.
```

## URL
[https://www.qualcomm.com/developer/software/something-something-v-2-dataset](https://www.qualcomm.com/developer/software/something-something-v-2-dataset)
