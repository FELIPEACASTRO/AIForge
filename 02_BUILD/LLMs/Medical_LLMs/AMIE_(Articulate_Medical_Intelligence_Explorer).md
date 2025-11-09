# AMIE (Articulate Medical Intelligence Explorer)

## Description

AMIE (Articulate Medical Intelligence Explorer) é um Grande Modelo de Linguagem (LLM) otimizado especificamente para o raciocínio diagnóstico clínico. Foi desenvolvido para gerar um Diagnóstico Diferencial (DDx) preciso, tanto de forma autônoma quanto como auxílio a clínicos. O modelo foi ajustado (fine-tuned) a partir do PaLM 2 (versão grande) usando uma mistura de tarefas médicas, incluindo resposta a perguntas, geração de diálogos clínicos e sumarização de notas de Registros Eletrônicos de Saúde (EHR). Sua otimização foca em contextos longos para aprimorar a capacidade de raciocínio de longo alcance e a compreensão contextual, essenciais para a complexidade dos casos médicos. O objetivo principal é melhorar a precisão diagnóstica em casos desafiadores e ampliar o acesso a expertise especializada.

## Statistics

**Precisão Top-10 (Standalone):** 59,1% (vs. 33,6% para clínicos não assistidos; P=0,04). **Precisão Top-1:** 29%. **Assistência Clínica:** A pontuação de qualidade do DDx foi maior para clínicos assistidos por AMIE (precisão top-10 de 51,7%) em comparação com clínicos sem assistência (36,1%) e com busca tradicional (44,4%; P=0,03). **Base:** PaLM 2 (versão grande). **Citação:** McDuff et al. (2025). Towards accurate differential diagnosis with large language models. Nature. [1]

## Features

Otimização para Raciocínio Clínico: Ajustado com tarefas médicas específicas (perguntas, diálogos, sumarização de EHR) para melhorar o raciocínio diagnóstico. Geração de DDx: Capacidade de gerar listas de Diagnóstico Diferencial (DDx) abrangentes e precisas. Desempenho Superior: Supera clínicos não assistidos e ferramentas de busca tradicionais na precisão do DDx. Assistência Clínica: Integrado em interfaces interativas para auxiliar médicos na formulação do DDx. Uso de Contexto Longo: Treinado para lidar com longos históricos clínicos e informações contextuais.

## Use Cases

**Suporte à Decisão Clínica:** Auxiliar médicos na formulação de Diagnósticos Diferenciais (DDx) para casos complexos e desafiadores. **Educação Médica:** Treinamento de estudantes e residentes, fornecendo raciocínio diagnóstico estruturado. **Triagem e Consulta Remota:** Geração de hipóteses diagnósticas iniciais em ambientes de telemedicina ou triagem de emergência. **Pesquisa:** Avaliação e comparação de diferentes abordagens de raciocínio clínico.

## Integration

A técnica de prompt mais eficaz para o AMIE e LLMs similares é o **Prompt de Raciocínio Clínico Estruturado** (Structured Clinical Reasoning Prompt), que imita o processo de pensamento de um médico.

**Exemplo de Prompt (Sintoma-para-Diagnóstico):**

```
Você é um especialista em diagnóstico médico. Sua tarefa é analisar o histórico do paciente e fornecer um Diagnóstico Diferencial (DDx) detalhado, seguido pelo diagnóstico mais provável.

**Passos de Raciocínio:**
1. **Coleta de Dados:** Liste os sintomas, histórico médico relevante e achados de exames (se houver).
2. **Análise de Padrões:** Identifique padrões e correlacione os achados com possíveis categorias de doenças.
3. **DDx:** Gere uma lista de 3 a 5 diagnósticos diferenciais plausíveis, justificando brevemente cada um.
4. **Diagnóstico Mais Provável:** Indique o diagnóstico mais provável e a justificativa para sua escolha.

**Histórico do Paciente:**
Paciente: Masculino, 50 anos.
Sintomas: Fadiga, perda de peso inexplicada (5kg em 3 meses), micção frequente (poliúria) e sede excessiva (polidipsia).
Histórico Médico: Sem histórico relevante, exceto hipertensão controlada.
```

**Guia de Integração:** O modelo AMIE foi integrado em uma interface interativa para medir seu impacto como assistente clínico, sugerindo que a integração ideal é via API em sistemas de Suporte à Decisão Clínica (CDSS) ou EHRs, utilizando prompts de múltiplas etapas (como CoT) para garantir a transparência do raciocínio.

## URL

https://www.nature.com/articles/s41586-025-08869-4