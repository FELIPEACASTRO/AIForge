# Arize AI - ML Observability and Monitoring

## Description

Arize AI is a unified **AI and Agent Engineering Platform** designed for observability and evaluation of Machine Learning (ML) models and AI Agents in production. Its unique value proposition is to close the loop between AI development and production, enabling a data-driven iteration cycle. The platform, known as **Arize AX**, empowers organizations to manage and improve AI offerings at scale, ensuring that production observability aligns with reliable evaluations. Arize AI serves both traditional ML and Computer Vision models and the growing field of Generative AI and LLMs.

## Statistics

The Arize AI platform processes large volumes of AI production data, with the following key metrics:
*   **1 Trillion** spans per month (agent and LLM tracing).
*   **50 Million** evaluations per month.
*   **5 Million** downloads per month (referring to Phoenix OSS, the open-source library for LLM Tracing and Evaluation).

## Features

The Arize AI platform (Arize AX) offers observability for Generative AI and ML & Computer Vision, covering three main pillars:

**1. Development:**
*   **Prompt Optimization:** Self-improving agents with automatic optimization based on evaluations and annotations.
*   **Playground Replay:** A tool for debugging and refining prompts.
*   **Prompt Serving and Management:** Centralized prompt management and fast serving of optimizations.

**2. Evaluation:**
*   **CI/CD Experiments:** Early detection of prompt and agent regressions through evaluation-driven CI/CD.
*   **LLM as a Judge:** Automatic, at-scale evaluation of prompts and agent actions.
*   **Human Annotation and Queues:** Management of labeling queues, production annotations, and creation of golden datasets.

**3. Observability:**
*   **Open Standard Tracing:** Agent and framework tracing powered by OpenTelemetry (OTEL).
*   **Online Evals:** Instant issue capture with AI evaluating AI.
*   **Monitoring and Dashboards:** Real-time AI monitoring with advanced analytical dashboards.

## Use Cases

Arize AI is used to ensure the reliability and performance of AI models and agents in production, covering both Generative AI and traditional ML and Computer Vision.

*   **AI Agents:** Real-time monitoring and debugging of AI agents, such as in fleet data analysis (Geotab) and travel bookings (Priceline).
*   **Bias Mitigation:** Examining credit scoring models to detect and mitigate bias, ensuring fair decisions free from discrimination.
*   **LLM Evals:** Applying real-time evaluations for large language models (LLMs), as in Bazaarvoice's examples, to measure metrics such as relevance, hallucination rate, and latency.
*   **Diverse Sectors:** Solutions for Computer Vision, Forecasting, Financial Services, and Manufacturing.
*   **Copilot Development:** Using the platform to develop, iterate on, and improve AI assistants (Copilots).

## Integration

Integration with Arize AI is primarily achieved through its SDKs and libraries, with a focus on Python.

**1. Arize AX (ML Observability) - Python SDK:**
The Python SDK is the primary tool for logging Machine Learning model data (predictions, actual labels, features, etc.) for monitoring.

```python
# Conceptual Example of Python SDK Usage (Arize AX)
from arize.api import Client

# Initialize the client with the API key and Hostname
arize_client = Client(
    space_key="YOUR_SPACE_KEY",
    api_key="YOUR_API_KEY",
    host="YOUR_HOST"
)

# Log prediction data for a model
arize_client.log_prediction(
    model_id="my-credit-model",
    prediction_id="unique-prediction-id",
    prediction_label="high_risk",
    actual_label="high_risk",
    features={"age": 35, "income": 50000},
    # ... other parameters such as embeddings, metrics, etc.
)
```

**2. Arize Phoenix (LLM Tracing and Evaluation) - Open Source:**
Phoenix is an open-source library for tracing and evaluating LLMs, which integrates with Arize AX.

```python
# Conceptual Example of Tracing with Phoenix (Python)
import phoenix as px
from phoenix.trace import Span

# Start the Phoenix tracer
px.start_session(
    project_name="my-llm-project",
    host="YOUR_PHOENIX_HOST" # Can be the local host or Arize AX
)

# Example of manual tracing of a span
with Span(name="response_generation", context={"prompt": "..."}) as span:
    # LLM call logic
    response = "LLM response"
    span.log_attributes({"response": response})
# Tracing also supports automatic integrations with frameworks such as OpenAI, LangChain, etc.
```

## URL

https://arize.com/