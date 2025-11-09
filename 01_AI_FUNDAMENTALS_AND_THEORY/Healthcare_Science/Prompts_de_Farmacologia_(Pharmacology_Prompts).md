# Prompts de Farmacologia (Pharmacology Prompts)

## Description
**Prompts de Farmacologia** referem-se à aplicação de técnicas de engenharia de prompt para interagir com Modelos de Linguagem Grande (LLMs) no domínio da farmacologia, descoberta de medicamentos, farmacovigilância, regulamentação farmacêutica e prática clínica. Essa abordagem visa otimizar a saída da IA para tarefas altamente especializadas, como análise de dados moleculares, simulação de interações medicamentosas, extração de informações de ensaios clínicos e garantia de conformidade regulatória em documentos farmacêuticos. A eficácia reside na capacidade de fornecer contexto clínico ou científico preciso, dados específicos (como estruturas SMILES ou resultados laboratoriais) e um formato de resposta rigoroso, garantindo que as saídas da IA sejam baseadas em evidências e alinhadas com as diretrizes clínicas e regulatórias mais recentes.

## Examples
```
1.  **Recomendação Farmacológica Clínica Baseada em Diretrizes**
    *   **Prompt:** "Atue como um farmacologista clínico. De acordo com as diretrizes da ADA de 2023, quais são os tratamentos farmacológicos recomendados para um paciente de 65 anos com diabetes tipo 2 e insuficiência cardíaca com fração de ejeção reduzida (FE < 40%)? Liste os medicamentos, seus mecanismos de ação e as contraindicações específicas para este perfil de paciente. Responda em formato de tabela Markdown."

2.  **Otimização de Moléculas para Descoberta de Medicamentos**
    *   **Prompt:** "Atue como um químico medicinal computacional. Dada a estrutura molecular SMILES: `CC(=O)Nc1ccc(cc1)O`, gere 5 análogos estruturalmente distintos que melhorem a afinidade pelo alvo [insira alvo, ex: COX-2] e mantenham a solubilidade em água. Forneça os novos SMILES e uma breve justificativa para cada alteração, focando na otimização de propriedades ADMET. Responda em formato JSON."

3.  **Extração Estruturada de Reações Adversas a Medicamentos (ADR)**
    *   **Prompt:** "Analise o seguinte texto de relato de caso [insira texto de prontuário ou artigo]. Extraia todas as Menções de Reações Adversas a Medicamentos (ADR), o medicamento associado e a gravidade (leve, moderada, grave). Use a seguinte estrutura JSON: `{"ADR": "...", "Medicamento": "...", "Gravidade": "..."}`. Se não houver ADRs, retorne um array vazio."

4.  **Análise de Interação Medicamentosa Complexa**
    *   **Prompt:** "Considere a combinação dos medicamentos [Medicamento A, ex: Varfarina] e [Medicamento B, ex: Fluconazol]. Descreva o mecanismo de interação farmacocinética (especificando a isoenzima CYP450 envolvida) e farmacodinâmica. Avalie o risco de interação (baixo, moderado, alto) e sugira uma estratégia de monitoramento laboratorial (ex: INR) para um paciente idoso com insuficiência hepática leve."

5.  **Revisão de Conformidade Regulatória de Bula**
    *   **Prompt:** "Você é um especialista em Assuntos Regulatórios. Revise o seguinte trecho de uma bula [insira texto] e verifique a consistência da terminologia e a conformidade com o Guia de Estilo da ANVISA (Agência Nacional de Vigilância Sanitária). Destaque quaisquer inconsistências de dosagem ou linguagem que possam levar a erros de uso pelo paciente. Sugira a reescrita do trecho para maior clareza e acessibilidade."

6.  **Criação de Estudo de Caso Educacional**
    *   **Prompt:** "Crie um estudo de caso simulado para estudantes de farmácia sobre o uso de [Nome do Medicamento, ex: Metformina] no tratamento de [Condição, ex: Síndrome do Ovário Policístico - SOP]. Inclua: a) Histórico detalhado do paciente, b) Farmacocinética e Farmacodinâmica relevantes, c) Plano de monitoramento terapêutico, d) 3 perguntas de múltipla escolha sobre o caso com gabarito. Mantenha a linguagem didática e profissional."

7.  **Simulação de Efeito de Ajuste de Dose em População Especial**
    *   **Prompt:** "Simule o impacto de uma redução de 50% na função renal (CrCl de 30 mL/min) na meia-vida e na concentração plasmática de estado estacionário (Css) do [Nome do Medicamento, ex: Digoxina], que é primariamente excretado por via renal. Explique o risco de toxicidade e calcule a dose ajustada recomendada para manter a Css dentro da faixa terapêutica (0.8-2.0 ng/mL). Justifique o cálculo com base nos princípios de farmacocinética."
```

## Best Practices
1.  **Estrutura C-D-T-F (Contexto, Dados, Tarefa, Formato):** Sempre estruture o prompt fornecendo um **Contexto** (ex: "Você é um farmacologista clínico"), os **Dados** a serem processados (ex: resultados de um ensaio), a **Tarefa** específica (ex: "Calcule a dose ajustada") e o **Formato** de saída desejado (ex: "Responda em JSON").
2.  **Alinhamento com Diretrizes:** Para prompts clínicos, instrua o LLM a basear sua resposta em diretrizes específicas e atualizadas (ex: "De acordo com as diretrizes da ADA de 2023...") para garantir a validade científica e clínica da saída.
3.  **Prompting com Exemplos (Few-Shot):** Para tarefas de alta precisão, como extração de Reações Adversas a Medicamentos (ADR) ou identificação de entidades, forneça exemplos de entrada e saída corretas para refinar o desempenho do modelo e corrigir erros.
4.  **Especificidade Química e Clínica:** Use terminologia técnica precisa. Em descoberta de medicamentos, utilize formatos padrão como SMILES para estruturas moleculares. Em clínica, inclua detalhes como idade, comorbidades, função renal e hepática para contextualizar a farmacocinética e farmacodinâmica.
5.  **Validação Cruzada:** Sempre inclua uma instrução para que o LLM cite as fontes ou referências (se possível) e trate a saída como uma sugestão que requer validação humana por um especialista qualificado.

## Use Cases
1.  **Revisão e Conformidade Regulatória:** Verificação de documentos (bulas, rótulos, SPCs) quanto à consistência de terminologia, precisão de dados e conformidade com diretrizes regulatórias (ex: FDA, EMA).
2.  **Descoberta e Otimização de Medicamentos:** Geração de análogos moleculares, previsão de propriedades ADMET (Absorção, Distribuição, Metabolismo, Excreção, Toxicidade) e identificação de alvos terapêuticos a partir de dados multi-ômicos.
3.  **Suporte à Decisão Clínica:** Geração de recomendações de tratamento farmacológico, ajuste de dosagem para populações especiais (renal, hepática) e análise de interações medicamentosas complexas.
4.  **Farmacovigilância e Segurança:** Extração e classificação de Reações Adversas a Medicamentos (ADR) de relatórios de caso ou literatura científica para monitoramento de segurança.
5.  **Educação e Treinamento:** Criação de estudos de caso simulados, guias de estudo sobre mecanismos de ação e simulação de entrevistas clínicas para estudantes e residentes.

## Pitfalls
1.  **Alucinação de Dados Clínicos:** O LLM pode gerar diretrizes, doses ou interações medicamentosas que parecem plausíveis, mas são factualmente incorretas ou desatualizadas. **Mitigação:** Exigir citação de fontes e validação humana.
2.  **Inconsistência Terminológica:** Falha em definir o vocabulário técnico (ex: usar "FE" sem especificar "Fração de Ejeção") pode levar a interpretações errôneas em contextos sensíveis. **Mitigação:** Fornecer um guia de estilo ou glossário no prompt.
3.  **Viés de Treinamento:** O modelo pode refletir vieses presentes nos dados de treinamento, resultando em recomendações que não são equitativas para todas as populações de pacientes. **Mitigação:** Incluir restrições de equidade e diversidade no prompt.
4.  **Ignorar o Contexto do Paciente:** Prompts muito genéricos sobre dosagem sem incluir o contexto completo do paciente (idade, comorbidades, função renal) podem levar a recomendações perigosas. **Mitigação:** Uso obrigatório da seção **Contexto** e **Dados** do prompt.
5.  **Dependência de Dados Privados:** Muitos casos de uso (ex: análise de prontuários) exigem o upload de dados sensíveis (PHI), o que é inviável ou inseguro com LLMs públicos. **Mitigação:** Usar apenas LLMs privados/seguros ou prompts que processem dados anonimizados/sintéticos.

## URL
[https://www.jmir.org/2025/1/e72644](https://www.jmir.org/2025/1/e72644)
