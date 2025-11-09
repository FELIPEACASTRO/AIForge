# MIMIC-III Clinical Database v1.4

## Description

O **MIMIC-III (Medical Information Mart for Intensive Care III)** é um extenso banco de dados de acesso livre e desidentificado, que contém informações de saúde relacionadas a mais de 40.000 pacientes que permaneceram em unidades de terapia intensiva (UTI) do Beth Israel Deaconess Medical Center entre 2001 e 2012 [1]. O conjunto de dados é uma base fundamental para pesquisas em informática médica e inteligência artificial na área da saúde. Ele suporta uma ampla gama de estudos analíticos, incluindo epidemiologia, melhoria de regras de decisão clínica e desenvolvimento de ferramentas eletrônicas [1]. O MIMIC-III é notável por sua disponibilidade gratuita para pesquisadores em todo o mundo, por abranger uma população grande e diversificada de pacientes de UTI e por conter dados altamente granulares, como sinais vitais, resultados laboratoriais e medicamentos [1]. Embora a versão mais recente seja de 2016 (v1.4), ele continua sendo amplamente utilizado em pesquisas recentes (2023-2025), muitas vezes em conjunto com seu sucessor, o MIMIC-IV [2] [3].

## Statistics

*   **Pacientes:** 46.520 pacientes únicos.
*   **Admissões Hospitalares:** 58.976 admissões únicas.
*   **Estadias na UTI:** 61.532 estadias únicas na UTI.
*   **Período de Coleta:** 2001 a 2012.
*   **Estrutura:** Banco de dados relacional com 26 tabelas.
*   **Tamanho:** Aproximadamente 40 GB (dados brutos) [1].
*   **Granularidade:** Contém dados de alta frequência (sinais vitais a cada hora) e dados esparsos (resultados laboratoriais, notas) [1].

## Features

O MIMIC-III é um banco de dados relacional composto por 26 tabelas, interligadas por identificadores como `SUBJECT_ID` (paciente único), `HADM_ID` (admissão hospitalar única) e `ICUSTAY_ID` (admissão única na UTI) [1]. Os dados são categorizados em:
*   **Dados Demográficos e Administrativos:** Informações do paciente, admissões, transferências e estadias na UTI.
*   **Eventos de Monitoramento:** Sinais vitais, eventos de monitoramento contínuo e eventos de entrada/saída (fluidos, medicamentos).
*   **Resultados Laboratoriais:** Resultados de exames de hematologia, química e microbiologia.
*   **Informações de Cobrança e Codificação:** Códigos ICD-9 (diagnósticos e procedimentos), DRG e CPT.
*   **Notas de Texto Livre:** Notas de cuidadores, resumos de alta e relatórios de imagens (desidentificados) [1].

**Técnicas de Feature Engineering Recentes (2023-2025):**
Pesquisas recentes focam em técnicas avançadas para extrair valor dos dados brutos do MIMIC-III:
1.  **Seleção de Características em Duas Camadas (Two-tier Feature Selection):** Utilizada para prever a mortalidade hospitalar, combinando métodos de seleção para otimizar modelos de *stacking* [4].
2.  **Extração de Características de Séries Temporais:** Uso de *Deep Learning* (como LSTMs) para capturar características temporais e espaciais de sinais de ECG e dados de sinais vitais para predição de desfechos clínicos [5] [6].
3.  **Extração de Características de Texto Livre:** Aplicação de modelos de *Deep Learning* e processamento de linguagem natural (NLP) para extrair características de notas médicas para classificação automatizada de códigos ICD [7].
4.  **Modelagem de Reforço Profundo (Deep Reinforcement Learning - DRL):** Utilizado para identificar estratégias de tratamento ideais, como no manejo da sepse, onde o DRL extrai as características de estado mais relevantes para a tomada de decisão [8].

## Use Cases

O MIMIC-III é a base para uma vasta gama de aplicações em pesquisa e desenvolvimento de IA na área da saúde:
*   **Predição de Mortalidade:** Modelos de *Machine Learning* (ML) e *Deep Learning* para prever a mortalidade hospitalar e em 30 dias em pacientes de UTI, incluindo aqueles com condições específicas como sepse e insuficiência cardíaca [4] [9].
*   **Previsão de Readmissão na UTI:** Desenvolvimento de modelos para identificar pacientes com alto risco de readmissão [2].
*   **Detecção Precoce de Condições Críticas:** Uso de modelos de *Ensemble* para melhorar a detecção precoce de condições como lesão renal aguda (LRA) em pacientes sépticos [10].
*   **Otimização de Tratamento:** Aplicação de *Deep Reinforcement Learning* para derivar estratégias de tratamento ideais, como no manejo da sepse [8].
*   **Classificação Automática de Códigos ICD:** Utilização de NLP em notas médicas para automatizar a codificação de diagnósticos e procedimentos [7].
*   **Análise de Sinais Fisiológicos:** Extração de características de sinais de ECG e sinais vitais para classificação de doenças como insuficiência cardíaca [5].

## Integration

O acesso ao MIMIC-III requer que o pesquisador conclua um curso de treinamento em proteção de dados humanos (por exemplo, o curso CITI) e assine um Acordo de Uso de Dados (DUA) no PhysioNet [1].

**Acesso e Integração:**
1.  **PhysioNet:** Após a credencialização, os arquivos de dados podem ser baixados diretamente do PhysioNet.
2.  **Plataformas em Nuvem:** O MIMIC-III está disponível no **Google Cloud Platform (GCP)** via BigQuery e no **Amazon Web Services (AWS)** via Athena, permitindo consultas e análises escaláveis diretamente na nuvem [1].

**Exemplo de Acesso (Conceitual - BigQuery/SQL):**
A integração tipicamente envolve consultas SQL complexas para unir as 26 tabelas e extrair as coortes e características desejadas.

```sql
-- Exemplo de consulta para obter a idade e o tempo de permanência na UTI
SELECT
    p.subject_id,
    EXTRACT(YEAR FROM p.dob) - EXTRACT(YEAR FROM a.admittime) AS age_at_admission,
    ROUND(CAST(ie.los AS NUMERIC), 2) AS icu_los_days
FROM
    physionet-data.mimiciii_clinical.patients p
INNER JOIN
    physionet-data.mimiciii_clinical.admissions a ON p.subject_id = a.subject_id
INNER JOIN
    physionet-data.mimiciii_clinical.icustays ie ON a.hadm_id = ie.hadm_id
WHERE
    a.admittime IS NOT NULL
LIMIT 10;
```
*Nota: Este é um exemplo conceitual de SQL para BigQuery. O acesso real requer credenciais e configuração na nuvem [1].*

## URL

https://physionet.org/content/mimiciii/