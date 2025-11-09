# Múltiplos (ADA, SFDA, SE_ASA)

## Description

Pesquisa abrangente sobre Adaptação de Domínio para Dados Médicos entre Hospitais (2023-2025), focando em três recursos chave: Adaptação de Domínio Adversarial Supervisionada (ADA) para classificação de raios-X de tórax, Adaptação de Domínio Livre de Fonte (SFDA) para segmentação de imagens de fundo de olho, e Adaptação de Domínio Não Supervisionada (UDA) SE_ASA para segmentação cardíaca em diferentes modalidades (MR/CT). Os resultados incluem métricas de desempenho, casos de uso e detalhes de integração, conforme solicitado.

## Statistics

ADA: Accuracy de 90.08%, AUC de 0.96 (CXR). SFDA: Dice médio de 91.74% (Drishti-GS) e 87.80% (RIM-ONE-r). SE_ASA: Dice médio de 74.6% (MR->CT) e 74.1% (CT->MR) para segmentação cardíaca.

## Features

Adaptação de Domínio Adversarial Supervisionada (ADA); Adaptação de Domínio Livre de Fonte (SFDA); Adaptação de Domínio Não Supervisionada (UDA) com Restrição de Entropia Seletiva e Alinhamento Semântico Adaptativo; Foco em generalização de modelos de IA para populações e instituições diversas; Preservação da privacidade dos dados.

## Use Cases

Classificação de raios-X de tórax em ambientes multi-institucionais; Segmentação de imagens de fundo de olho; Segmentação de estruturas cardíacas em diferentes modalidades de imagem (MR/CT).

## Integration

Instruções de integração variam por recurso: ADA (código disponível sob solicitação, dataset no Kaggle); SFDA (metodologia detalhada para implementação); SE_ASA (código-fonte PyTorch no GitHub com instruções de treinamento/teste).

## URL

Múltiplos (Nature, MDPI, GitHub)