# Engenharia de Prompt e LLMs para Geração de Relatórios Médicos

## Description

Pesquisa abrangente sobre as mais recentes técnicas de engenharia de prompt e Modelos de Linguagem Grande (LLMs) aplicados à geração de relatórios médicos (2023-2025). O foco está em metodologias de prompt (Zero-shot, CoT, GKP) e modelos de ponta como MRG-LLM e Med-PaLM 2, com detalhes sobre arquitetura, métricas de desempenho e exemplos práticos de prompts.

## Statistics

**MRG-LLM (2025):** Alcançou desempenho de última geração em conjuntos de dados de raios-X de tórax (IU X-ray e MIMIC-CXR), com o modelo *Prompt-wise* atingindo BLEU-4 de 0.170 e ROUGE-L de 0.332 no IU X-ray. **Med-PaLM 2 (2023-2025):** Pontua até 86.5% no conjunto de dados MedQA, melhorando em mais de 19% em relação ao Med-PaLM original. A versão multimodal (Med-PaLM M) demonstrou melhoria de mais de 8% nos escores de geração de relatórios de raios-X de tórax.

## Features

Técnicas de Prompting Clínico (Zero-shot, Few-shot, Chain-of-Thought, Generated Knowledge, Meta-prompting); LLM Multimodal Específico (MRG-LLM) para Radiologia; LLM de Alto Desempenho (Med-PaLM 2) para Questões Médicas e Geração de Relatórios.

## Use Cases

Automatização da redação de relatórios de imagem (Radiologia); Auxílio à decisão clínica e diagnóstico diferencial; Resumo de histórico médico e planos de tratamento; Geração de materiais educativos para pacientes.

## Integration

**Exemplo de Prompt para Redação de Relatório (Zero-shot):** "Aja como um médico experiente e redija um relatório médico completo para o paciente [Nome do Paciente], com base nos seguintes dados: [Insira dados do paciente, histórico, achados de exame físico, resultados de exames e diagnóstico]." **Exemplo de Prompt CoT:** "Liste os diagnósticos diferenciais para dor no peito, especificando o raciocínio passo a passo e os testes necessários para cada um."

## URL

https://www.jmir.org/2025/1/e72644; https://arxiv.org/html/2506.15477v1; https://sites.research.google/med-palm/