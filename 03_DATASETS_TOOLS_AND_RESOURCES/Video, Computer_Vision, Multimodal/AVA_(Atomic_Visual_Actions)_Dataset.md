# AVA (Atomic Visual Actions) Dataset

## Description
O **AVA (Atomic Visual Actions)** é um conjunto de dados de vídeo para anotações audiovisuais que visa melhorar a compreensão da atividade humana. O dataset principal, **AVA Actions v2.2**, anota densamente 80 ações visuais atômicas em 430 clipes de filmes de 15 minutos, onde as ações são localizadas no espaço e no tempo. O projeto também inclui o **AVA-Kinetics** (uma combinação com o Kinetics-700 para localização de ações em uma variedade mais ampla de cenas), **AVA ActiveSpeaker** (associação de atividade de fala com um rosto visível) e **AVA Speech** (anotação de atividade de fala baseada em áudio), tornando-o um recurso multimodal robusto.

## Statistics
- **AVA Actions v2.2:** 430 vídeos (235 treino, 64 validação, 131 teste), cada um com 15 minutos anotados em intervalos de 1 segundo. Total de 1.62M rótulos de ação.
- **AVA-Kinetics v1.0:** 430 vídeos do AVA v2.2 + 238k vídeos do Kinetics-700.
- **AVA ActiveSpeaker v1.0:** 3.65 milhões de quadros rotulados em aproximadamente 39 mil trilhas faciais.
- **AVA Speech v1.0:** Aproximadamente 46 mil segmentos rotulados, abrangendo 45 horas de dados.

## Features
- **Localização Espaço-Temporal de Ações:** As ações são localizadas no espaço (caixas delimitadoras) e no tempo (intervalos de 1 segundo), permitindo uma análise detalhada da dinâmica da atividade humana.
- **Ações Atômicas:** Anotação de 80 ações visuais atômicas (por exemplo, "ficar em pé", "apertar a mão", "falar").
- **Múltiplos Rótulos por Pessoa:** Permite que uma pessoa tenha múltiplos rótulos de ação simultaneamente.
- **Multimodal:** Inclui sub-datasets para análise de fala e falante ativo, integrando visão e áudio.

## Use Cases
- **Reconhecimento e Localização de Ações em Vídeos:** Treinamento de modelos para identificar e localizar ações humanas em tempo real.
- **Detecção de Falante Ativo e Análise de Atividade de Fala:** Aplicações em sistemas de conferência, segurança e interação humano-computador.
- **Pesquisa em Visão Computacional:** Desenvolvimento de novos algoritmos para compreensão de atividades humanas e modelagem de interações sociais.
- **Transfer Learning:** Uso do AVA-Kinetics para expandir a generalização de modelos de localização de ação.

## Integration
O dataset é fornecido como arquivos CSV contendo as anotações. Os vídeos originais são identificados por IDs do YouTube e devem ser baixados separadamente (o que pode exigir ferramentas de terceiros, como o `youtube-dl` ou scripts específicos). O formato CSV de anotação inclui: `video_id`, `middle_frame_timestamp`, `person_box` (x1, y1, x2, y2 normalizados), `action_id`, e `person_id`. Os arquivos de anotação (CSV) estão disponíveis para download direto na página oficial. O código de avaliação (Frame-mAP) está disponível no GitHub do ActivityNet.

## URL
[https://research.google.com/ava/](https://research.google.com/ava/)
