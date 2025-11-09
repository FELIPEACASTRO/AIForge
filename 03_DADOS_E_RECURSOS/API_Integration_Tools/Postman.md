# Postman

## Description

O Postman é a plataforma líder mundial para desenvolvimento e colaboração de APIs, oferecendo um conjunto completo de ferramentas para cada estágio do ciclo de vida da API. Sua proposta de valor única reside em ser uma solução "API-first" completa, que vai além de um simples cliente REST, permitindo a gestão de especificações, documentação, testes automatizados, monitoramento e colaboração em escala. É a ferramenta de escolha para equipes que buscam uma abordagem unificada e escalável para o desenvolvimento de APIs.

## Statistics

* **Adoção:** Mais de 30 milhões de desenvolvedores e 500.000 empresas, incluindo 98% das empresas da Fortune 500, utilizam o Postman.
* **Tendência API-First:** O Postman é um dos principais impulsionadores da abordagem API-First, com 82% das organizações adotando algum nível dessa abordagem.
* **Relatório Anual:** Publica o "State of the API Report", uma das principais referências sobre tendências e uso de APIs no mundo.

## Features

* **Workspace Colaborativo:** Permite que equipes trabalhem juntas em coleções de APIs, ambientes e documentação.
* **Coleções de APIs:** Agrupamento lógico de requisições e testes, que podem ser compartilhados e executados em lote.
* **Testes Automatizados:** Criação de scripts de teste (em JavaScript) para validar respostas de APIs, integrados a pipelines de CI/CD.
* **Monitoramento de APIs:** Monitoramento contínuo da performance e disponibilidade das APIs em diferentes regiões.
* **Documentação Automática:** Geração de documentação interativa a partir das coleções.
* **Mock Servers:** Simulação de endpoints de API para desenvolvimento e testes paralelos.
* **Suporte a Múltiplos Protocolos:** Suporte a REST, SOAP, GraphQL, gRPC e WebSockets.

## Use Cases

* **Desenvolvimento e Debugging de APIs:** Envio rápido de requisições e inspeção de respostas para acelerar o desenvolvimento.
* **Testes de Regressão e Funcionais:** Criação de suítes de testes robustas para garantir a qualidade da API antes do deploy.
* **Onboarding de Desenvolvedores:** Compartilhamento de coleções de APIs para que novos membros da equipe ou parceiros possam começar a usar a API rapidamente.
* **Governança de APIs:** Aplicação de padrões e diretrizes de design de APIs em toda a organização.

## Integration

O Postman facilita a integração com diversas ferramentas e linguagens, sendo capaz de gerar snippets de código para a requisição atual em mais de 20 linguagens e frameworks.

**Exemplo de Integração (Python - `requests`):**

```python
import requests
import json

url = "https://api.exemplo.com/dados"

payload = json.dumps({
  "chave": "valor"
})
headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer SEU_TOKEN'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)
```

**Integração com CI/CD:** Utiliza o Newman (executor de linha de comando do Postman) para rodar coleções de testes em ambientes de integração contínua (Jenkins, GitLab CI, GitHub Actions).

## URL

https://www.postman.com/