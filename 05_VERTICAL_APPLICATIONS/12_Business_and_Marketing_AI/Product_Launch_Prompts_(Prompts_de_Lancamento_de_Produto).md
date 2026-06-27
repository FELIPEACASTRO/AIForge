# Product Launch Prompts (Prompts de Lançamento de Produto)

## Description
**Prompts de Lançamento de Produto** são uma categoria especializada de engenharia de prompt focada em alavancar Modelos de Linguagem Grande (LLMs) para automatizar, acelerar e aprimorar todas as fases de uma estratégia de lançamento de produto (Go-to-Market - GTM). Em vez de prompts genéricos, eles são estruturados para extrair saídas de marketing, vendas, relações públicas e produto altamente contextuais e acionáveis.

A técnica se baseia em fornecer à IA um **briefing de produto** detalhado (público, USP, objetivos) e solicitar entregáveis específicos do lançamento, como planos de conteúdo, cópia de funil de vendas, kits de imprensa e análises competitivas. Isso transforma a IA de um gerador de texto em um **estrategista de lançamento virtual**, permitindo que equipes de marketing e produto criem campanhas complexas em uma fração do tempo tradicional. A eficácia reside na capacidade de simular especialistas de domínio (copywriters, analistas, estrategistas) e solicitar resultados que se encaixem diretamente em um plano de lançamento estruturado.

## Examples
```
**1. Blueprint de Estratégia de Lançamento Completa**

```
Você é um Estrategista de Produto de elite e consultor de Go-to-Market. Estou lançando um novo produto chamado [Nome do Produto] no nicho de [Indústria/Nicho]. Nosso público-alvo são [Público-Alvo] e nosso principal diferencial é [USP Principal]. Crie um plano de lançamento completo cobrindo: 1) Pesquisa de audiência pré-lançamento, 2) Calendário de conteúdo de 30 dias, 3) Estratégia de influenciadores/parceiros, e 4) Métricas-chave para rastrear durante a semana de lançamento. Apresente o plano em formato de tabela com ações diárias, ferramentas a serem usadas e KPIs esperados.
```

**2. Matriz de Posicionamento e Mensagens**

```
Você é um Estrategista de Marca Sênior. Ajude-me a desenvolver uma estrutura de posicionamento e mensagens para o meu produto: [Nome do Produto]. O produto resolve [Problema] para [Público-Alvo]. Crie: 1) Declaração de Posicionamento, 2) Pitch de Elevador, 3) 3 opções de Slogan, e 4) 5 Pilares de Mensagens com pontos de prova. Apresente em um esboço estruturado com sugestões de tom/estilo para diferentes canais (LinkedIn, X, Landing Page).
```

**3. Copywriting de Funil de Lançamento (Sequência de E-mail)**

```
Você é um copywriter de resposta direta de primeira linha. Crie uma sequência de e-mail de lançamento de funil completo para [Nome do Produto] projetada para [Tipo de Público]. Inclua: 1) E-mail de Teaser (3 dias antes), 2) E-mail de Anúncio (Dia do Lançamento), e 3) E-mail de Fechamento (24h antes do fim da oferta). Cada e-mail deve seguir meu tom: [Descreva o tom, ex: casual-educacional, ousado-inspirador]. Use gatilhos emocionais e fluxo lógico para maximizar as conversões.
```

**4. Gerador de Ângulo de Relações Públicas e Kit de Imprensa**

```
Você é um Estrategista de Relações Públicas para lançamentos de tecnologia. Estou lançando [Nome do Produto] em [Data de Lançamento]. Gere um esboço de kit de imprensa, incluindo: 1) Título e estrutura do comunicado de imprensa, 2) 3 ângulos de história de mídia (tendência, fundador, inovação), 3) Resumo da biografia do fundador, e 4) E-mail de pitch de contato para jornalistas. Use um tom profissional, mas incisivo, adequado para a mídia de tecnologia/startup.
```

**5. Plano de Engajamento e Buzz Pré-Lançamento**

```
Você é um especialista em crescimento de comunidade e estratégia de lançamento social. Quero criar um buzz massivo 30 dias antes de lançar [Nome do Produto]. Desenvolva um calendário de conteúdo e engajamento de 30 dias de pré-lançamento. Inclua: 1) Tópicos de Storytelling, 2) Postagens de construção de comunidade, 3) Colaborações com criadores, e 4) Métricas para medir a viralidade. Apresente tudo em uma tabela semanal estruturada com tipos de postagem, plataforma e resultados esperados.
```
```

## Best Practices
**1. Forneça Contexto Completo (O Método do Briefing)**: Sempre inclua o máximo de detalhes possível: nome do produto, público-alvo, proposta de valor única (USP), tom de voz desejado, e o objetivo específico do lançamento (ex: vendas, inscrições, buzz). O modelo de IA não é um especialista em marketing sem a sua entrada.

**2. Use a Estrutura de Funil (Tease, Announce, Close)**: Estruture seus prompts para cobrir todas as fases do lançamento: pré-lançamento (geração de buzz), lançamento (anúncio e vendas) e pós-lançamento (retenção e feedback).

**3. Atribua uma Persona de Especialista (Role-Playing)**: Comece o prompt definindo o papel da IA (ex: "Você é um Copywriter de Resposta Direta de Nível Sênior", "Você é um Estrategista de Relações Públicas para Startups de Tecnologia"). Isso melhora a qualidade e o foco da saída.

**4. Solicite Formatos Estruturados**: Peça a saída em formatos fáceis de usar, como tabelas, listas numeradas ou JSON. Isso facilita a revisão e a implementação (ex: "Apresente o plano em formato de tabela com ações diárias e KPIs").

**5. Itere e Refine (Prompt Chain)**: Não espere o resultado perfeito na primeira tentativa. Use a saída de um prompt como entrada para o próximo. Por exemplo, use um prompt para definir a **Mensagem** e outro para gerar a **Cópia** com base nessa mensagem.

## Use Cases
**1. Geração de Estratégia Go-to-Market (GTM)**: Criar rapidamente um plano de lançamento de alto nível, incluindo cronogramas, canais e KPIs, em vez de começar do zero.

**2. Desenvolvimento de Mensagens e Posicionamento**: Definir a Proposta de Valor Única (USP), o Pitch de Elevador e os Pilares de Mensagens para garantir a consistência em todos os materiais de marketing.

**3. Criação de Conteúdo de Funil de Vendas**: Gerar rascunhos de cópia para páginas de destino, sequências de e-mail (teaser, anúncio, fechamento), scripts de vídeo e ganchos de anúncios pagos.

**4. Análise Competitiva Rápida**: Simular uma análise de inteligência competitiva para identificar lacunas de mercado e ângulos de diferenciação para o novo produto.

**5. Automação de Relações Públicas (PR)**: Criar esboços de comunicados de imprensa, kits de imprensa e e-mails de pitch para jornalistas, acelerando o processo de divulgação de mídia.

## Pitfalls
**1. Confiança Excessiva na IA (O Mito do 'Prompt Único')**: Acreditar que um único prompt pode gerar uma estratégia de lançamento completa e perfeita. A IA é uma ferramenta de aceleração, não um substituto para a revisão humana, validação de mercado e tomada de decisão estratégica.

**2. Falta de Contexto Específico**: Usar prompts genéricos sem fornecer detalhes cruciais sobre o produto, público-alvo, concorrentes e diferenciais. Isso resulta em saídas genéricas e inutilizáveis.

**3. Ignorar a Voz da Marca**: Não especificar o tom de voz e a personalidade da marca. A IA pode gerar cópia tecnicamente correta, mas que não ressoa com a identidade da empresa.

**4. Não Iterar ou Refinar**: Aceitar a primeira saída da IA. Os melhores resultados vêm de um processo iterativo, onde a saída de um prompt é refinada ou usada como entrada para o próximo.

**5. Viés de Confirmação**: Usar a IA apenas para confirmar ideias existentes em vez de explorar ângulos novos ou desafiadores (ex: "Peça à IA para analisar a fraqueza de sua ideia de produto antes de lançar").

## URL
[https://medium.com/@slakhyani20/10-new-chatgpt-prompts-for-product-launches-fb7c1e7c27a4](https://medium.com/@slakhyani20/10-new-chatgpt-prompts-for-product-launches-fb7c1e7c27a4)
