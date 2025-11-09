# Documentation Prompts

## Description
"Documentation Prompts" (Prompts de Documentação) são instruções estruturadas e detalhadas fornecidas a um modelo de linguagem grande (LLM) com o objetivo específico de gerar, revisar ou aprimorar documentação técnica, manuais de usuário, artigos de base de conhecimento ou FAQs. A técnica se baseia em fornecer ao LLM os quatro elementos cruciais para a criação de documentação de alta qualidade: **Personagem** (o papel que o LLM deve assumir, como "Escritor Técnico Sênior"), **Intenção** (o objetivo da documentação, como "explicar o processo de instalação"), **Cenário** (o contexto específico, como "para usuários iniciantes do software X"), e **Entrega** (o formato e o estilo de saída, como "em formato Markdown com tom amigável"). O uso eficaz desta técnica transforma o LLM em um assistente de escrita técnica, acelerando drasticamente o ciclo de criação de conteúdo.

## Examples
```
1. **Personagem:** "Aja como um Engenheiro de Suporte de Nível 3." **Intenção:** "Crie um guia de solução de problemas." **Cenário:** "O usuário está recebendo o erro 'HTTP 503 Service Unavailable' ao tentar acessar a API." **Entrega:** "Gere um artigo de base de conhecimento em Markdown com uma lista numerada de 5 passos para diagnóstico e resolução, começando com a verificação do status do servidor."

2. **Personagem:** "Você é um redator de manuais de usuário com foco em simplicidade." **Intenção:** "Escreva a seção 'Primeiros Passos'." **Cenário:** "O usuário acabou de instalar o aplicativo 'Finanças Rápidas' e precisa configurar a primeira conta bancária." **Entrega:** "Produza um texto conciso, com no máximo 3 parágrafos, usando um tom encorajador e destacando o botão 'Adicionar Conta' em negrito."

3. **Personagem:** "Assuma o papel de um especialista em segurança de software." **Intenção:** "Revise e aprimore a documentação de segurança." **Cenário:** "O trecho a seguir descreve a autenticação via OAuth 2.0: [TRECHO A SER REVISADO]." **Entrega:** "Reescreva o trecho para garantir que a terminologia esteja 100% precisa e que as melhores práticas de segurança (como o uso de tokens de curta duração) sejam enfatizadas."

4. **Personagem:** "Seja um redator de FAQs para um produto SaaS." **Intenção:** "Gere 10 perguntas e respostas frequentes." **Cenário:** "O produto é uma ferramenta de gerenciamento de projetos baseada em Kanban. As dúvidas mais comuns são sobre preços, integrações e limites de usuários." **Entrega:** "Crie uma lista de 10 FAQs, com respostas diretas e concisas, usando a voz da marca (profissional e prestativa)."

5. **Personagem:** "Aja como um desenvolvedor sênior." **Intenção:** "Documente a função de API `calculate_tax(amount, country_code)`." **Cenário:** "A função aceita um float e uma string de 2 letras e retorna um float. A documentação deve seguir o padrão JSDoc." **Entrega:** "Gere a documentação completa da função, incluindo descrição, parâmetros de entrada, tipo de retorno e um exemplo de uso em Python."

6. **Personagem:** "Você é um tradutor técnico fluente em Português e Inglês." **Intenção:** "Traduza e localize um manual de instruções." **Cenário:** "Traduza o seguinte texto do Inglês para o Português do Brasil, mantendo o tom formal: [TEXTO EM INGLÊS]." **Entrega:** "Forneça a tradução em Português, garantindo que termos técnicos como 'firmware' e 'interface' sejam mantidos ou traduzidos corretamente para o contexto brasileiro."
```

## Best Practices
**Seja Específico e Estruturado:** Use os quatro elementos-chave (Personagem, Intenção, Cenário e Entrega) para estruturar seu prompt. **Defina o Formato de Saída:** Especifique o formato desejado (Markdown, HTML, JSON, estilo de manual, etc.) e a estrutura (títulos, subtítulos, listas). **Forneça Contexto e Exemplos:** Inclua informações de fundo relevantes e, se possível, um exemplo de documentação existente para que o LLM imite o tom e o estilo. **Mantenha a Clareza e a Concisão:** Evite ambiguidades. Cada instrução deve ser clara sobre o que se espera. **Adicione Palavras-Chave:** Para documentação técnica ou de SEO, inclua uma lista de palavras-chave a serem incorporadas. **Revisão Humana é Obrigatória:** Sempre revise e edite o conteúdo gerado pela IA para garantir precisão técnica, tom de voz e ausência de erros factuais.

## Use Cases
**Criação de Manuais de Usuário:** Geração rápida de guias passo a passo para novos produtos ou recursos. **Desenvolvimento de Base de Conhecimento (Knowledge Base):** Produção em massa de artigos de suporte e FAQs para reduzir o volume de tickets de suporte. **Documentação de API e Código:** Criação de documentação técnica padronizada (como JSDoc, Sphinx ou OpenAPI) para funções, classes e endpoints de API. **Localização e Tradução:** Tradução e adaptação de documentação existente para diferentes idiomas e contextos culturais. **Geração de Tutoriais e Guias:** Criação de conteúdo educacional e tutoriais para onboarding de clientes. **Revisão e Padronização:** Uso do prompt para revisar documentação existente, garantindo consistência de tom, estilo e precisão técnica.

## Pitfalls
**Falta de Contexto:** Não fornecer o Personagem, Intenção, Cenário ou Entrega leva a documentação genérica e inútil. **Dependência Excessiva:** Confiar cegamente na saída da IA sem revisão humana pode resultar em erros técnicos graves ou informações desatualizadas. **Prompts Vagos:** Usar prompts como "Escreva sobre o produto X" sem especificar o público-alvo, o objetivo ou o formato. **Ignorar o Tom de Voz:** Não definir o tom (formal, amigável, técnico) resulta em documentação inconsistente com a marca. **Ausência de Exemplos:** Não fornecer um exemplo de documentação existente impede o LLM de replicar o estilo e a estrutura de forma ideal. **Foco Apenas no Conteúdo:** Esquecer de especificar a estrutura de entrega (títulos, listas, negritos) pode gerar um bloco de texto difícil de ler.

## URL
[https://betterdocs.co/ai-prompt-writing-for-documentation/](https://betterdocs.co/ai-prompt-writing-for-documentation/)
