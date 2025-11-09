# VoxCeleb (Speaker Recognition)

## Description
O **VoxCeleb** é um conjunto de dados de áudio e visual em grande escala, projetado para tarefas de reconhecimento de locutor (speaker recognition) e verificação de locutor (speaker verification) em cenários "in the wild" (não controlados). O dataset é composto por clipes de fala extraídos de vídeos de entrevistas de celebridades carregados no YouTube. A versão mais recente e abrangente é o **VoxCeleb2**, que contém mais de 1 milhão de enunciados de 6.112 celebridades. O dataset é notável por sua diversidade, abrangendo uma ampla gama de etnias, sotaques, profissões e idades, e por capturar a fala em condições reais, incluindo ruído de fundo, risadas e sobreposição de fala. Embora os links de download diretos do site oficial tenham sido removidos por questões de privacidade, o dataset continua sendo o padrão ouro para pesquisa na área, com desafios anuais (VoxSRC) e diversas implementações de terceiros para download e uso.

## Statistics
**VoxCeleb1:**
- **Locutores:** 1.251 celebridades
- **Enunciados:** > 150.000
- **Duração:** Não especificada (mas menor que VoxCeleb2)

**VoxCeleb2 (Versão mais utilizada):**
- **Locutores:** 6.112 celebridades
- **Enunciados:** > 1.092.009 (Desenvolvimento) + 36.237 (Teste) = **> 1.128.246**
- **Duração Total:** **> 2.000 horas**
- **Vídeos:** 145.569 (Desenvolvimento) + 4.911 (Teste) = **150.480**
- **Versões:** VoxCeleb1, VoxCeleb2, e desafios anuais (VoxSRC) que utilizam e expandem o dataset.

## Features
- **Áudio-Visual:** Contém dados de áudio e vídeo, permitindo o desenvolvimento de modelos multimodais.
- **"In the Wild":** Coletado de vídeos do YouTube, o que garante variabilidade de pose, iluminação, ruído de fundo e qualidade de áudio.
- **Grande Escala:** O VoxCeleb2 possui mais de 1 milhão de enunciados e 6.112 identidades.
- **Diversidade:** Abrange uma ampla gama de etnias, sotaques, profissões e idades.
- **Foco em Celebridades:** A identidade de cada locutor é uma celebridade pública, facilitando a coleta de dados.

## Use Cases
- **Reconhecimento de Locutor (Speaker Identification):** Identificar quem está falando a partir de uma amostra de voz.
- **Verificação de Locutor (Speaker Verification):** Confirmar se a identidade reivindicada por um locutor corresponde à sua voz.
- **Reconhecimento de Emoção pela Voz:** Embora não seja o foco principal, o dataset tem sido usado para estudos de emoção (ex: EmoVoxCeleb).
- **Processamento de Fala Multimodal:** Pesquisa em fusão de informações de áudio e vídeo (face-tracks) para melhorar a robustez dos sistemas.
- **Desafios de Pesquisa (VoxSRC):** Utilizado como base para o VoxCeleb Speaker Recognition Challenge, um dos principais benchmarks da área.

## Integration
Devido à remoção dos links diretos do site oficial por questões de privacidade, a integração do VoxCeleb é tipicamente realizada através de scripts de terceiros ou repositórios espelho.

1.  **Scripts de Download:** O método mais comum é utilizar scripts Python/Shell disponíveis em repositórios como o GitHub (ex: `clovaai/voxceleb_trainer` ou `walkoncross/voxceleb2-download`) que automatizam o processo de download dos vídeos do YouTube (com base nos metadados de URL e timestamp) e a extração dos segmentos de áudio/vídeo.
2.  **Repositórios de Terceiros:** Algumas plataformas como o Academic Torrents ou o Hugging Face (ex: `ProgramComputer/voxceleb`) oferecem versões pré-processadas ou links para o dataset completo, embora o usuário deva sempre verificar a licença e a integridade dos dados.
3.  **Protocolos de Avaliação:** Para tarefas de verificação de locutor, os protocolos de avaliação (listas de pares de teste) são fornecidos no site oficial e são essenciais para garantir a reprodutibilidade dos resultados.

*Nota: É necessário aceitar os Termos e Condições e estar ciente das questões de privacidade e licenciamento Creative Commons Attribution-ShareAlike 4.0 International.*

## URL
[https://www.robots.ox.ac.uk/~vgg/data/voxceleb/](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
