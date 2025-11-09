# Neurology Prompts (Engenharia de Prompt em Neurologia)

## Description
A Engenharia de Prompt em Neurologia refere-se à arte e ciência de criar instruções (prompts) otimizadas para Modelos de Linguagem Grande (LLMs) com o objetivo de auxiliar em tarefas clínicas, de pesquisa e administrativas dentro do campo da Neurologia. Não é uma técnica de prompt singular, mas sim a aplicação de técnicas avançadas como **RAG (Retrieval-Augmented Generation)**, **definição de função (Role Definition)** e **estrutura de saída (Output Schema)** para garantir que os LLMs forneçam respostas clinicamente relevantes, precisas e seguras. Estudos recentes (2025) demonstram sua eficácia na geração de resumos de consultas e na previsão de risco em ambientes de emergência, embora alertem para a necessidade de validação rigorosa e supervisão humana devido às limitações no raciocínio clínico sutil [1] [2] [3].

## Examples
```
1. **Geração de Resumo Clínico Estruturado (RAG-like)**

```
**Função:** Você é um assistente de documentação neurológica. Sua tarefa é gerar um resumo conciso de alta fidelidade para o prontuário eletrônico.

**Contexto (RAG):** [INSERIR AQUI: Histórico do paciente, resultados de exames de imagem e laboratório, nota de enfermagem e exame neurológico inicial.]

**Instrução:** Analise o contexto e gere um resumo com as seguintes seções e restrições:
1. **Queixa Principal (QP):** Máximo 1 frase.
2. **Achados Chave do Exame:** 3 a 5 pontos principais.
3. **Diagnóstico Diferencial (DD):** Lista de 3 DDs mais prováveis.
4. **Plano de Conduta Sugerido:** Lista de 3 ações imediatas.

**Formato de Saída:** JSON, com as chaves 'QP', 'Achados', 'DD' e 'Conduta'.
```

2. **Suporte a Diagnóstico Diferencial para Cefaleia**

```
**Função:** Você é um neurologista consultor especializado em cefaleias. O paciente apresenta [INSERIR: Idade, Sexo, Duração, Localização, Intensidade (escala 1-10), Sintomas Associados (náusea, fotofobia, aura)].

**Instrução:** Com base nos dados, forneça:
1. O diagnóstico primário mais provável.
2. Dois diagnósticos diferenciais que não podem ser descartados.
3. Uma lista de 3 'Red Flags' (sinais de alerta) que exigiriam investigação imediata.

**Restrição:** A resposta deve ser didática e justificar cada diagnóstico com base nos sintomas fornecidos.
```

3. **Previsão de Risco de Admissão em Emergência**

```
**Função:** Você é um sistema de triagem de risco neurológico (Neuro-Copilot AI).

**Dados de Entrada:** [INSERIR: Pontuação NIHSS, Idade, Pressão Arterial, Glicemia, Tempo de Sintomas, Presença de Comorbidades (ex: Fibrilação Atrial)].

**Instrução:** Calcule a probabilidade de:
1. Necessidade de admissão hospitalar (Baixa, Média, Alta).
2. Risco de mortalidade em 48 horas (%).

**Restrição:** Se a probabilidade de admissão for 'Alta', adicione a frase: 'REQUER AVALIAÇÃO IMEDIATA POR NEUROLOGISTA'. Se os dados estiverem incompletos, responda: 'DADOS INSUFICIENTES PARA PREVISÃO'.
```

4. **Interpretação Simplificada de Neuroimagem (Para Paciente)**

```
**Função:** Você é um comunicador médico. Sua tarefa é traduzir o relatório de ressonância magnética para uma linguagem que um paciente com ensino médio possa entender.

**Relatório Original:** [INSERIR: Trecho do laudo, ex: 'Múltiplas lesões hiperintensas em T2 e FLAIR, periventriculares e justacorticais, compatíveis com doença desmielinizante.']

**Instrução:** Explique o achado em 3 parágrafos curtos. Use analogias se necessário. Evite jargões técnicos. Mantenha um tom tranquilizador e informativo.
```

5. **Extração de Dados Estruturados de Nota de Progresso**

```
**Função:** Você é um extrator de dados para um sistema de qualidade.

**Nota de Progresso:** [INSERIR: Nota de evolução do dia, ex: 'Paciente com Parkinson, sem alteração na dose de levodopa. Apresenta bradicinesia leve, mas tremor de repouso controlado. Sem quedas. Próxima consulta em 3 meses.']

**Instrução:** Extraia os seguintes campos e formate-os em JSON:
- **Doença:**
- **Medicação Chave:**
- **Dose Alterada (Sim/Não):**
- **Sintoma Dominante:**
- **Próximo Follow-up (Meses):**
```

6. **Sugestão de Protocolo de Investigação para Neuropatia**

```
**Função:** Você é um especialista em investigação de neuropatias periféricas.

**Dados:** [INSERIR: História clínica (DM, etilismo, quimioterapia), Achados do Exame Físico (Padrão de déficit sensitivo/motor), Resultado do EMG (ex: Neuropatia axonal sensitivo-motora crônica).]\n\n**Instrução:** Sugira um protocolo de investigação laboratorial de segunda linha (após exames básicos) em formato de lista numerada, priorizando as causas mais prováveis.
```

7. **Criação de Cenário de Simulação Clínica para Treinamento**

```
**Função:** Você é um designer de currículo médico.

**Tópico:** Crise Epiléptica Tônico-Clônica Generalizada.

**Instrução:** Crie um cenário de simulação clínica para residentes de neurologia, incluindo:
1. **Apresentação do Paciente:** (Idade, Sexo, História Breve).
2. **Ações Iniciais do Residente (Checklist):** (5 itens).
3. **Ponto Crítico de Decisão:** (Ex: Quando administrar a segunda dose de benzodiazepínico).
4. **Desfecho da Simulação:** (Breve descrição).
```

8. **Revisão de Literatura Focada em Mecanismo de Ação**

```
**Função:** Você é um pesquisador sênior.

**Fármaco:** [INSERIR: Nome do fármaco, ex: Fingolimod]

**Instrução:** Revise a literatura dos últimos 5 anos e descreva o mecanismo de ação do fármaco no contexto da Esclerose Múltipla. Concentre-se em:
1. Alvos moleculares primários.
2. Efeitos imunológicos e não imunológicos.
3. Impacto na barreira hematoencefálica.

**Restrição:** Use referências (cite o autor e ano) e limite a resposta a 300 palavras.
```
```

## Best Practices
A eficácia dos Neurology Prompts depende da sua estruturação e da integração de dados contextuais. As melhores práticas incluem:

*   **Definição de Função e Persona:** Sempre instrua o LLM a agir como um profissional de saúde específico (ex: 'Você é um neurologista consultor'), elevando a qualidade e o tom da resposta.
*   **RAG (Contextualização):** Para tarefas clínicas, o LLM deve ser aumentado com dados do paciente (histórico, exames, notas). Isso mitiga a alucinação e garante a relevância clínica [1].
*   **Estrutura de Saída Rígida:** Exija formatos estruturados (JSON, tabelas, listas numeradas) para facilitar a análise, a integração com Prontuários Eletrônicos (EMRs) e a automação de fluxos de trabalho.
*   **Restrições de Segurança:** Inclua instruções para o LLM recusar ou solicitar mais informações se os dados de entrada forem insuficientes, ambíguos ou se a tarefa exceder o escopo de um assistente de IA (ex: 'Não forneça diagnóstico final, apenas sugestões').
*   **Tom e Linguagem:** Especifique o tom (clínico, conciso, didático) e o público-alvo (colega médico, paciente, pesquisador) para otimizar a comunicação.

## Use Cases
Os casos de uso primários para a Engenharia de Prompt em Neurologia se concentram em aumentar a eficiência e a precisão clínica:

*   **Otimização de Documentação:** Geração automática de resumos de consultas, notas de progresso e cartas de alta, reduzindo a carga administrativa [1].
*   **Suporte à Decisão Clínica (CDS):** Auxílio na formulação de diagnósticos diferenciais, sugestão de planos de investigação e tratamento, e interpretação de dados complexos (ex: EEG, EMG) [2].
*   **Triage e Previsão de Risco:** Uso de modelos baseados em LLM para prever desfechos críticos, como a necessidade de admissão ou risco de mortalidade em curto prazo, especialmente em ambientes de emergência [3].
*   **Educação e Treinamento:** Criação de cenários de simulação clínica, geração de perguntas de múltipla escolha e tradução de jargões médicos para a educação de pacientes e estudantes.
*   **Pesquisa:** Análise e sumarização de grandes volumes de literatura científica e extração de dados estruturados de artigos para metanálises.

## Pitfalls
A aplicação de LLMs na neurologia apresenta riscos significativos que devem ser mitigados pela Engenharia de Prompt:

*   **Alucinação Clínica (Inacurácia):** O risco de o LLM gerar informações clinicamente falsas ou inventar referências é alto, podendo levar a erros de diagnóstico ou tratamento. Isso é agravado pela falta de raciocínio sutil em casos complexos [2].
*   **Superprescrição de Testes:** LLMs sem treinamento específico tendem a sugerir mais testes diagnósticos do que o necessário, aumentando custos e sobrecarga do sistema de saúde [2].
*   **Viés e Iniquidade:** Se os dados de treinamento forem enviesados (ex: sub-representação de certas populações), os prompts podem perpetuar disparidades no cuidado, levando a recomendações inadequadas para grupos minoritários.
*   **Dependência Excessiva:** A confiança cega nas saídas do LLM, sem validação por um profissional humano, é o maior risco. O LLM deve ser visto como um assistente, não como um substituto para o julgamento clínico.
*   **Vazamento de Dados Sensíveis:** A inclusão de dados de pacientes no prompt (mesmo que anonimizados) requer protocolos de segurança rigorosos para evitar a exposição de Informações de Saúde Protegidas (PHI).

## URL
[https://www.nature.com/articles/s41598-025-22769-7](https://www.nature.com/articles/s41598-025-22769-7)
