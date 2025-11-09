# ChestX-ray14 Dataset (NIH Chest X-ray Dataset)

## Description

O ChestX-ray14 é um dos maiores e mais utilizados conjuntos de dados de radiografias de tórax em grande escala. Foi lançado pelo National Institutes of Health (NIH) e contém 112.120 imagens de raios-X frontais de 30.805 pacientes únicos, anotadas para a presença de até 14 diferentes patologias torácicas. As anotações foram geradas automaticamente usando técnicas de mineração de texto em relatórios de radiologia, o que o torna um conjunto de dados 'fracamente rotulado'. É amplamente utilizado como benchmark para o desenvolvimento de modelos de aprendizado profundo para classificação e localização de doenças torácicas.

## Statistics

Consiste em 112.120 imagens de raios-X de tórax frontais de 30.805 pacientes únicos. As imagens originais estão no formato PNG com resolução de 1024 x 1024. O conjunto de dados totaliza aproximadamente 45 GB.

## Features

As imagens são rotuladas para a presença de até 14 patologias torácicas: Atelectasia, Cardiomegalia, Derrame (Effusion), Infiltração (Infiltration), Massa (Mass), Nódulo (Nodule), Pneumonia, Pneumotórax, Consolidação (Consolidation), Edema, Enfisema, Fibrose, Espessamento Pleural e Hérnia. Também inclui metadados como idade, sexo e posição de visualização (AP/PA).

## Use Cases

Classificação multi-rótulo de doenças torácicas, detecção de pneumonia, desenvolvimento de modelos de IA para priorização de casos urgentes, estudos de fairness e viés em IA médica, e extração de características radiômicas (radiomics) para diagnóstico assistido por computador. Pesquisas recentes (2023-2025) continuam a utilizá-lo para validação de novos modelos de *deep learning* e *foundation models* em radiografia de tórax.

## Integration

O conjunto de dados pode ser acessado e baixado diretamente do NIH (via Box.com) ou através de plataformas como Kaggle e Hugging Face, que fornecem versões pré-processadas e ferramentas de acesso simplificadas. A integração geralmente envolve o uso de bibliotecas Python como PyTorch ou TensorFlow, com carregadores de dados personalizados para lidar com a estrutura multi-rótulo e os metadados. Exemplo de acesso ao arquivo de metadados via Pandas (Kaggle): `import pandas as pd\ndf = pd.read_csv('/kaggle/input/nih-chest-xrays/Data_Entry_2017.csv')\nprint(df.head())`

## URL

https://nihcc.app.box.com/v/ChestXray-NIHCC