# OK-VQA (Outside Knowledge Visual Question Answering)

## Description
O OK-VQA (Outside Knowledge Visual Question Answering) é um dataset de Perguntas e Respostas Visuais (VQA) que exige que os modelos utilizem **conhecimento externo** (fora do conteúdo da imagem) para responder às perguntas. Diferentemente dos datasets VQA tradicionais, onde as respostas podem ser inferidas apenas a partir da imagem, o OK-VQA contém mais de 14.000 perguntas de resposta aberta que foram manualmente filtradas para garantir a necessidade de conhecimento externo (por exemplo, da Wikipédia). O dataset foi projetado para impulsionar a pesquisa em modelos que podem integrar informações visuais e textuais de fontes de conhecimento externas. Uma versão aprimorada, o **OK-VQA v2.0**, foi lançada em 2023, corrigindo e removendo 41,4% e 10,6% do dataset original, respectivamente, para melhorar a qualidade. O dataset é frequentemente usado em conjunto com o A-OKVQA (Augmented OK-VQA), que fornece racionalizações para as respostas.

## Statistics
- **Amostras:** 14.055 perguntas de resposta aberta.
- **Respostas:** 5 respostas de referência (ground truth) por pergunta.
- **Versões:**
    - **v1.0 (2019):** Versão original.
    - **v1.1 (2020):** Atualização com melhorias no método de *word stemming* nas respostas.
    - **v2.0 (2023):** Versão aprimorada com correções e remoções para maior qualidade.
- **Tamanho:** O tamanho dos arquivos de anotações (JSON) é pequeno (alguns MB). O dataset completo, incluindo as imagens COCO, é significativamente maior (as imagens COCO 2014 têm ~25 GB). A versão no Hugging Face (apenas anotações) tem cerca de **832 MB** (lmms-lab/OK-VQA).

## Features
- **Requisito de Conhecimento Externo:** A característica principal é a necessidade de conhecimento fora da imagem para responder às perguntas.
- **Perguntas de Resposta Aberta:** Mais de 14.000 perguntas que exigem respostas de formato livre.
- **Múltiplas Respostas de Referência:** Cada pergunta possui 5 respostas de referência (ground truth) coletadas por humanos.
- **Categorias de Conhecimento:** As perguntas são categorizadas em 10 tipos de conhecimento externo, como Pessoas e Vida Cotidiana, Ciência e Tecnologia, História, Geografia, Esportes e Recreação, etc.
- **Baseado em COCO:** As imagens são provenientes do dataset COCO (Common Objects in Context).

## Use Cases
- **Avaliação de Modelos VQA:** Serve como um benchmark desafiador para modelos de Perguntas e Respostas Visuais que precisam de raciocínio baseado em conhecimento.
- **Pesquisa em Conhecimento Externo:** Desenvolvimento de métodos para integrar bases de conhecimento externas (como a Wikipédia) em modelos de Visão Computacional e Processamento de Linguagem Natural.
- **Raciocínio Multimodal:** Treinamento de modelos para realizar raciocínio complexo que combina informações visuais e fatos do mundo real.
- **Transferência de Conhecimento:** Estudo da capacidade dos modelos de transferir conhecimento de grandes modelos de linguagem para tarefas de VQA.

## Integration
O dataset OK-VQA pode ser baixado diretamente do site oficial do Allen AI, onde os arquivos de anotações (perguntas e respostas) são fornecidos em formato JSON. As imagens correspondentes são do dataset COCO.

**Passos para Integração:**
1.  **Baixar Anotações:** Baixar os arquivos JSON de anotações de treinamento e teste (v1.1 ou v2.0, se disponível) do site oficial.
2.  **Obter Imagens:** As imagens não estão incluídas no download das anotações e devem ser obtidas do dataset **COCO 2014** (conjuntos de treinamento e validação).
3.  **Processamento:** Usar as anotações JSON para mapear as perguntas e respostas às imagens correspondentes do COCO.
4.  **Alternativa (Hugging Face):** O dataset também está disponível em plataformas como o Hugging Face, onde pode ser carregado diretamente usando a biblioteca `datasets` do Python:
    ```python
    from datasets import load_dataset
    # Para a versão do Hugging Face (pode não incluir as imagens COCO)
    dataset = load_dataset("lmms-lab/OK-VQA")
    ```

## URL
[https://okvqa.allenai.org/](https://okvqa.allenai.org/)
