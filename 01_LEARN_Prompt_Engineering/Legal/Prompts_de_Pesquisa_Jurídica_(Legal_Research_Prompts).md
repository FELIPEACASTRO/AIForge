# Prompts de Pesquisa Jurídica (Legal Research Prompts)

## Description
A Engenharia de Prompts Jurídicos é o processo de criar instruções precisas e detalhadas para sistemas de Inteligência Artificial Generativa (GenAI) com o objetivo de obter respostas precisas, relevantes e úteis para tarefas legais, como pesquisa de jurisprudência, análise de documentos, elaboração de peças e sumarização de regulamentos. É uma habilidade crucial para advogados que buscam maximizar a eficiência e a precisão das ferramentas de IA, mitigando riscos como as "alucinações" (respostas factualmente incorretas) [1] [2]. A precisão na formulação do prompt é o diferencial entre um resultado genérico e um produto de trabalho jurídico acionável e verificável [3]. O foco principal é a clareza, a definição de um papel (role-playing) e a exigência de uma estrutura de raciocínio lógico (como o formato IRAC) para garantir a verificação humana obrigatória do produto final [1].

## Examples
```
**1. Pesquisa de Jurisprudência e Sumarização (IRAC)**

> **Prompt:** "Aja como um advogado sênior de Direito do Consumidor. Analise o caso [Nome do Caso/Número do Processo] e aplique o formato IRAC (Issue, Rule, Application, Conclusion). O foco deve ser na **regra de direito** utilizada para determinar a responsabilidade por danos morais em casos de negativação indevida. Cite o artigo específico do Código de Defesa do Consumidor (CDC) e resuma a decisão em até 300 palavras."

**2. Análise de Cláusula Contratual**

> **Prompt:** "Você é o consultor jurídico de uma startup de SaaS. Revise a Cláusula 7 ('Indenização') do [Anexo o Contrato/Cole a Cláusula] e responda: A cláusula é favorável ao meu cliente (o Licenciante)? Quais são os três maiores riscos de responsabilidade que ela impõe? Reescreva a cláusula em linguagem simples e não jurídica, mantendo o mesmo efeito legal, para que a equipe de vendas possa entender."

**3. Elaboração de Rascunho de Peça Processual**

> **Prompt:** "Aja como um advogado cível com 10 anos de experiência. Elabore um rascunho de petição inicial para uma ação de despejo por falta de pagamento. Inclua as seções de 'Fatos', 'Fundamentação Jurídica' (citando a Lei do Inquilinato e o Código de Processo Civil) e 'Pedidos'. O valor da causa é R$ [Valor]. O réu está inadimplente há 4 meses. Mantenha um tom formal e persuasivo."

**4. Sumarização de Regulamento Complexo**

> **Prompt:** "Você é um especialista em conformidade regulatória (Compliance). Analise o [Regulamento Específico, ex: Lei Geral de Proteção de Dados - LGPD]. Resuma as **cinco principais obrigações** de um Controlador de Dados. Para cada obrigação, cite o artigo exato da LGPD e forneça um exemplo prático de como uma empresa de médio porte deve cumprir essa obrigação. O output deve ser uma tabela concisa."

**5. Geração de Perguntas para Entrevista**

> **Prompt:** "Aja como um advogado trabalhista. Estou prestes a entrevistar um ex-funcionário que alega assédio moral. Gere uma lista de 10 perguntas estratégicas e não indutivas para a entrevista, focadas em coletar fatos objetivos e evitar a confissão de culpa. Inclua uma breve justificativa legal para cada pergunta."

**6. Comparação de Jurisprudências**

> **Prompt:** "Compare e contraste as decisões recentes do Superior Tribunal de Justiça (STJ) sobre a aplicação da teoria da perda de uma chance em casos de erro médico. Identifique pelo menos dois acórdãos divergentes (se houver) e destaque os critérios que o STJ tem utilizado para quantificar a indenização. O resultado deve ser uma análise comparativa, não apenas um resumo."
```

## Best Practices
**1. Defina o Papel da IA (Role-Playing):** Comece o prompt instruindo a IA a agir como um profissional específico (ex: "Aja como um advogado cível sênior", "Você é o consultor jurídico de uma startup de alto crescimento") para garantir que a resposta adote o tom e a perspectiva corretos [1] [4].

**2. Utilize o Formato IRAC (Issue, Rule, Application, Conclusion):** Peça explicitamente à IA para estruturar sua análise jurídica usando o formato IRAC. Isso permite que o usuário diagnostique facilmente a cadeia de raciocínio da IA e verifique a precisão de cada etapa [1].

**3. Forneça Contexto Amplo, mas Não Direcione a Resposta:** Inclua o máximo de contexto e fatos relevantes possível antes de fazer a pergunta. No entanto, formule perguntas abertas para evitar influenciar o resultado (o que é conhecido como "leading the witness" - direcionar a testemunha) [1].

**4. Exija Citações Verificáveis:** Sempre solicite que a IA cite a seção específica da lei, regulamento ou jurisprudência. Mesmo que as citações da IA possam estar incorretas (alucinações), elas fornecem um ponto de partida para a verificação humana, que é obrigatória [1] [5].

**5. Itere e Refine o Prompt:** Se a resposta inicial não for satisfatória, use a própria IA para refinar o prompt. Pergunte: "Com base na sua resposta anterior, quais conceitos-chave posso ajustar ou refinar para mudar o resultado?" [1].

**6. Use a Linguagem Simples para o Output:** Peça à IA para resumir ou explicar conceitos complexos em "linguagem simples e não jurídica" (plain, non-legal language) para facilitar a comunicação com clientes ou partes não jurídicas [1].

## Use Cases
nan

## Pitfalls
**1. Alucinações e Falsas Citações (Hallucinations):** O erro mais grave. A IA pode inventar casos, leis ou citações que parecem reais, mas são factualmente incorretas. **Solução:** Nunca confie em uma citação sem verificá-la na fonte primária [2] [5].

**2. Prompts Excessivamente Genéricos:** Pedir "elabore uma petição" ou "resuma a lei" sem fornecer contexto, papel, jurisdição ou objetivo específico. Isso resulta em respostas vagas e inúteis [4].

**3. Vazamento de Informações Confidenciais:** Inserir dados sensíveis de clientes ou casos em modelos de IA de propósito geral (como o ChatGPT público) que não garantem a confidencialidade. **Solução:** Utilize apenas ferramentas de IA jurídicas seguras ou modelos privados/locais para dados confidenciais [1].

**4. Falha na Definição do Papel (Role):** Não especificar o papel da IA (ex: advogado, juiz, consultor). A IA pode assumir um papel de "assistente pessoal" e fornecer respostas que priorizam a satisfação do usuário em vez da precisão legal [1].

**5. Confiança Cega na Resposta:** Tratar a saída da IA como um produto final. A IA deve ser vista como um "advogado júnior" ou assistente, cujo trabalho deve ser sempre revisado, verificado e validado por um profissional humano [1] [2].

## URL
[https://www.lsuite.co/blog/mastering-ai-legal-prompts](https://www.lsuite.co/blog/mastering-ai-legal-prompts)
