# IBM watsonx.ai (Evolution of IBM Watson Studio)

## Description

IBM watsonx.ai is the evolution of IBM Watson Studio, an integrated, end-to-end AI development studio designed to simplify and scale the building and deployment of AI models, including foundation models and generative AI. Its unique value proposition lies in being a unified platform that provides access to trusted foundation models (such as IBM Granite, third-party models, and open-source models via Model Gateway), allowing companies to focus on applying AI for business outcomes. It delivers a collaborative development experience, with or without code, for the entire AI lifecycle, from data preparation to deployment and monitoring, and can be deployed in hybrid cloud environments (SaaS or self-hosted on IBM Cloud Pak for Data).

## Statistics

**Customer Outcomes (watsonx.ai):** AddAI saw 50% fewer unanswered customer service inquiries; Silver Egg Technology anticipates a 75% faster hiring process; UHCW NHS Trust served 700 more patients weekly without adding staff; Blendow Group saw 90% less time needed to summarize and analyze documents. **Support Metrics (G2):** watsonx.ai outperforms Watson Studio with a score of 8.8 in support quality.

## Features

Model Gateway (access to foundation models such as Granite, third-party, and open source); Full AI Lifecycle Management; Developer AI Toolkit (SDKs, APIs, agentic workflows, RAG frameworks); Content and Code Generation; Knowledge Management (with RAG templates); Insight Extraction and Forecasting; support for Hybrid environments (Cloud and on-premises).

## Use Cases

**Customer Service:** Reduction of unanswered inquiries and improved satisfaction. **Human Resources:** Acceleration of the hiring process. **Healthcare:** Optimization of patient care capacity. **Document Analysis:** Efficient summarization and analysis of large volumes of documents. **AI Agent Development:** Creation of AI assistants and agents to automate business processes.

## Integration

Integration is done primarily through the **IBM watsonx.ai Python SDK** (`ibm-watsonx-ai`) and REST APIs.

**Text Inference Example (Python SDK):**
```python
from ibm_watsonx_ai.client import Client
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

# 1. Client Configuration
# Replace 'YOUR_API_KEY' and 'YOUR_PROJECT_ID'
client = Client(
    url="https://us-south.ml.cloud.ibm.com",
    api_key="YOUR_API_KEY",
    project_id="YOUR_PROJECT_ID"
)

# 2. Parameter Definition
parameters = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 50,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.REPETITION_PENALTY: 1
}

# 3. Prompt and Model
prompt = "Explain the concept of generative AI in one sentence."
model_id = ModelTypes.GRANITE_13B_INSTRUCT

# 4. Text Generation
response = client.model.generate(
    model_id=model_id,
    prompt=prompt,
    params=parameters
)

# 5. Result
generated_text = response['results'][0]['generated_text']
# print(generated_text)
```

## URL

https://www.ibm.com/products/watsonx-ai