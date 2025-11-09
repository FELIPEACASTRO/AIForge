# GRU-D e TBAL (Modelos RNN para Monitoramento de Pacientes)

## Description

Pesquisa abrangente sobre Redes Neurais Recorrentes (RNNs) para Monitoramento de Pacientes, focada em publicações recentes (2023-2025), resultando na identificação e detalhamento de dois modelos de destaque: GRU-D e TBAL.

**GRU-D (Gated Recurrent Unit with Decay) para Vigilância de Íleo Pós-operatório**
*   **Descrição:** Modelo RNN com Mecanismo de Decaimento para avaliação de risco em tempo real de íleo pós-operatório (IPO) em cirurgia colorretal. Ideal para dados clínicos longitudinais esparsos, incorporando imputação de dados faltantes que considera o tempo desde a última observação.
*   **Estatísticas:** Validado em 7349 registros da Mayo Clinic. AUROC em transferência multi-fonte melhorou em até 2.6%, demonstrando robusta transferibilidade entre diferentes sistemas EHR (Centricity e EPIC).
*   **Casos de Uso:** Vigilância de IPO em tempo real, avaliação de risco de outras complicações pós-operatórias (infecção, sangramento), monitoramento contínuo em UTI.
*   **URL:** https://www.nature.com/articles/s43856-025-01053-9

**TBAL (Time-aware Bidirectional Attention-based Long Short-Term Memory)**
*   **Descrição:** Modelo RNN baseado em LSTM Bidirecional com Atenção Sensível ao Tempo para previsão dinâmica e em tempo real do risco de mortalidade em pacientes de UTI. Projetado para lidar com a natureza irregular e longitudinal dos dados de Prontuários Eletrônicos (EMR).
*   **Estatísticas:** Validado em 176.344 internações em UTI (MIMIC-IV e eICU-CRD). AUROC para previsão estática (12h a 1 dia) de 95.9 (MIMIC-IV) e 93.3 (eICU-CRD).
*   **Casos de Uso:** Previsão de mortalidade em tempo real em UTI, sistema de alerta precoce para deterioração clínica, suporte à decisão clínica.
*   **URL:** https://www.jmir.org/2025/1/e69293

## Statistics

**GRU-D:** Validado em 7349 registros de cirurgia colorretal em três locais da Mayo Clinic. O modelo demonstrou desempenho superior aos modelos atemporais (regressão logística e random forest) nas horas pós-cirúrgicas. Em transferência 'brute-force', a AUROC (Area Under the Receiver Operating Characteristic) diminuiu em no máximo 5%. A transferência de instância multi-fonte resultou em uma melhoria de até 2.6% na AUROC e um intervalo de confiança 86% mais estreito, demonstrando robusta transferibilidade.
**TBAL:** Validado em 176.344 internações em UTI (MIMIC-IV e eICU-CRD). Previsão estática (12h a 1 dia): AUROC de 95.9 (MIMIC-IV) e 93.3 (eICU-CRD). Previsão dinâmica: AUROC de 93.6 (MIMIC-IV) e 91.9 (eICU-CRD). Alta recall para casos positivos (82.6% e 79.1%). A validação cruzada entre bases de dados confirmou a generalizabilidade (AUROCs de 81.3 e 76.1).

## Features

**GRU-D:** Avaliação de risco dinâmica em tempo real; Lida com extrema esparsidade de dados (ex: 72.2% dos laboratórios e 26.9% dos sinais vitais faltantes em 24h pós-cirurgia); Mecanismo de decaimento para imputação de dados faltantes; Robustez e transferibilidade entre diferentes sistemas EHR (Centricity e EPIC) e locais hospitalares.
**TBAL:** Previsão de risco dinâmica e em tempo real (atualizada a cada hora); Arquitetura LSTM Bidirecional para capturar dependências temporais em ambas as direções; Mecanismo de Atenção Sensível ao Tempo para priorizar informações mais relevantes; Lida com amostragem irregular e dados faltantes; Solução robusta, interpretável (usando Integrated Gradients) e generalizável.

## Use Cases

**GRU-D:** Vigilância de íleo pós-operatório em tempo real; Avaliação de risco de outras complicações pós-operatórias (infecção superficial, infecção de ferida, sangramento); Monitoramento contínuo de pacientes em unidades de terapia intensiva (UTI) ou pós-cirurgia.
**TBAL:** Previsão de mortalidade em tempo real em pacientes de UTI; Sistema de alerta precoce para deterioração clínica; Suporte à decisão clínica para intervenções oportunas.

## Integration

**GRU-D:** O modelo é uma arquitetura RNN baseada em GRU. A implementação requer a adaptação do código original de Che et al. (2018) para o contexto clínico específico, utilizando dados longitudinais de sinais vitais, resultados laboratoriais e outras variáveis clínicas. A integração em sistemas de monitoramento requer a capacidade de processar sequências de dados clínicos em tempo real e aplicar o mecanismo de decaimento para as variáveis esparsas.
**TBAL:** O modelo é baseado em uma arquitetura LSTM Bidirecional e requer dados longitudinais de EMR (sinais vitais, laboratórios, medicamentos). Foi treinado e validado usando os bancos de dados públicos MIMIC-IV e eICU-CRD. A implementação requer a adaptação do código para processar sequências de dados clínicos com consciência do tempo e aplicar o mecanismo de atenção. O artigo menciona que detalhes adicionais estão no Apêndice Multimídia 1.

## URL

https://www.nature.com/articles/s43856-025-01053-9; https://www.jmir.org/2025/1/e69293