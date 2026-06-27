# Habitat (Embodied AI) / Habitat-Matterport 3D Dataset (HM3D)

## Description
Habitat é uma plataforma de simulação de alto desempenho para pesquisa em Inteligência Artificial Incorporada (Embodied AI), desenvolvida pelo Meta AI. O principal dataset associado é o **Habitat-Matterport 3D Dataset (HM3D)**, a maior coleção de espaços internos 3D de alta resolução (gêmeos digitais) para treinar agentes incorporados (robôs virtuais e assistentes egocêntricos) em ambientes fotorrealistas e eficientes. A plataforma é composta por: (i) **Habitat-Sim**, um simulador 3D de alto desempenho com física, e (ii) **Habitat-Lab**, uma biblioteca de alto nível para desenvolvimento e treinamento de agentes. A versão mais recente é o **Habitat 3.0** (2023), que foca na co-habitação de humanos, avatares e robôs.

## Statistics
**HM3D:** 1.000 cenas 3D em escala de construção. Espaço navegável total de **112.500 m²**. Espaço total de chão de **365.420 m²**. Versões notáveis incluem o **Habitat 3.0** (2023) e o **HM3D-Sem** (2023), que adiciona anotações semânticas. O artigo original do HM3D é de 2021.

## Features
**Plataforma Habitat:** Simulação 3D fotorrealista e eficiente, com suporte a física (via Bullet), sensores configuráveis (RGB-D, egomotion) e robôs descritos via URDF (Fetch, Franka, AlienGo). O Habitat-Sim atinge milhares de quadros por segundo (FPS). **Dataset HM3D:** 1.000 reconstruções 3D em escala de construção de ambientes reais (residenciais, comerciais, cívicos). Cada cena é composta por malhas 3D texturizadas e metadados detalhados (classificação de revisor, número de andares/quartos, espaço navegável, complexidade de navegação e desordem de cena). O HM3D-Sem (2023) adiciona anotações semânticas densas.

## Use Cases
Treinamento e avaliação de agentes de IA incorporada (Embodied AI), como robôs domésticos e assistentes egocêntricos. Tarefas de navegação autônoma (ObjectNav, ImageNav), manipulação de objetos, rearranjo de ambientes (Habitat 2.0), e interação humano-agente (Habitat 3.0). O dataset HM3D é um benchmark para pesquisa em percepção ativa, planejamento de longo prazo e aprendizado por interação em ambientes 3D realistas.

## Integration
O acesso ao dataset HM3D é gratuito para fins de pesquisa acadêmica e não comercial. O download é feito através do site da Matterport, exigindo os seguintes passos: 1. Criar uma conta gratuita na Matterport. 2. Acessar as **Developer Tools** nas configurações da conta. 3. Solicitar acesso ao **Habitat - Matterport 3D Research Dataset** e preencher o formulário. Após a aprovação, o dataset pode ser baixado manualmente ou programaticamente usando o utilitário `datasets_download` do Habitat-Sim. O Habitat-Lab é instalado via pip ou conda.

## URL
[https://aihabitat.org/](https://aihabitat.org/)
