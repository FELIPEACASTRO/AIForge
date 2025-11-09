# Rasa

## Description

Rasa é um framework de código aberto para aprendizado de máquina que permite aos desenvolvedores criar assistentes contextuais baseados em texto e voz. Ele se concentra em diálogos complexos e conversacionais, separando o entendimento da linguagem natural (NLU) do gerenciamento de diálogo.

## Statistics

Estrelas no GitHub: 20.828. Uma das plataformas de IA conversacional de código aberto mais populares e amplamente adotadas.

## Features

Arquitetura modular (NLU e Core), gerenciamento de diálogo contextual, suporte a múltiplos canais, integração com modelos de linguagem grandes (LLMs), extensibilidade via Ações Personalizadas (Custom Actions).

## Use Cases

Atendimento ao cliente automatizado, assistentes internos para funcionários, bots de vendas e marketing, assistentes de voz e IVR conversacional.

## Integration

Pode ser integrado a canais como Slack, Telegram, Facebook Messenger, Twilio e websites. Exemplo de integração com Python para rodar o servidor Rasa e enviar mensagens:\n\n```python\nimport requests\n\nrasa_server_url = 'http://localhost:5005/webhooks/rest/webhook'\n\nmessage = {\n    'sender': 'user_id',\n    'message': 'Olá, preciso de ajuda com minha conta.'\n}\n\nresponse = requests.post(rasa_server_url, json=message)\n\nif response.status_code == 200:\n    for message in response.json():\n        print(f\"Bot: {message.get('text')}\")\nelse:\n    print(f\"Erro ao se comunicar com o servidor Rasa: {response.status_code}\")\n```

## URL

https://rasa.com/