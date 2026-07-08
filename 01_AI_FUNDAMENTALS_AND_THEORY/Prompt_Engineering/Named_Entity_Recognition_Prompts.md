# Named Entity Recognition Prompts

## Description
Named Entity Recognition (NER) Prompts are Prompt Engineering techniques focused on guiding Large Language Models (LLMs) to identify and classify named entities (such as people, organizations, locations, dates, etc.) in unstructured text. Instead of relying on traditional Machine Learning (ML) models trained on large volumes of labeled data, the prompt-based approach leverages the reasoning and context-comprehension capabilities of LLMs to perform the NER task, often specifying the desired output format (e.g., JSON, XML, or BIO notation). This technique takes advantage of the reasoning ability and inherent knowledge of LLMs to perform information extraction tasks with high accuracy and flexibility, making it a powerful alternative to traditional NER models.

## Examples
```
1. **Simple Zero-Shot (General Extraction)**
   ```
   Instruction: Extract all named entities (PERSON, ORGANIZATION, LOCATION) from the following text and list them in the format: [Entity]: [Type].
   Text: "Dr. Ana Silva, from the University of São Paulo (USP), presented her research in Berlin last week."
   ```

2. **Few-Shot for the Financial Domain**
   ```
   Instruction: You are a financial analyst. Extract the entities (COMPANY, VALUE, CURRENCY, DATE) from the text.
   Example 1: "Petrobras (COMPANY) announced a profit of 10 billion (VALUE) reais (CURRENCY) in the third quarter of 2023 (DATE)."
   Example 2: "Apple (COMPANY) reached a market value of 3 trillion (VALUE) dollars (CURRENCY) in January 2024 (DATE)."
   Text: "Banco do Brasil reported 5% growth in its 2024 balance sheet, totaling 15.5 billion reais."
   ```

3. **Structured Extraction (JSON)**
   ```
   Instruction: Extract the entities (PRODUCT, QUANTITY, UNIT) from the shopping list and return the output strictly in JSON format, following the schema: [{"entidade": "...", "tipo": "...", "valor": "..."}].
   Text: "Buy 3 kilos of rice, 1 dozen eggs, and 500 grams of mozzarella cheese."
   ```

4. **Function Calling Simulation (System Prompt)**
   ```
   System Prompt:
   You are a data extraction assistant. Your only function is to call the `extract_clinical_entities` tool with the correct arguments.
   Tool: `extract_clinical_entities(doenca: str, sintoma: str, medicamento: str)`
   
   User Prompt:
   "Patient João was diagnosed with pneumonia and is taking Amoxicillin to treat the fever and persistent cough."
   ```

5. **Domain-Specific Prompt (Legal)**
   ```
   Instruction: Identify and classify the entities (PARTY, COURT, LAW, DATE) in the following legal excerpt. Be precise.
   Text: "The decision was handed down by the Supremo Tribunal Federal (STF) on May 15, 2024, in favor of the Claimant Maria da Penha, based on Article 5 of the Federal Constitution."
   ```

6. **Chain-of-Thought (CoT) for Disambiguation**
   ```
   Instruction: Analyze the text and extract the entities (PERSON, LOCATION). Before providing the final answer, use CoT reasoning to justify the classification of ambiguous entities.
   Text: "Paris, the capital of France, is a common name. Paris Hilton, on the other hand, is a celebrity."
   ```

7. **Extraction with BIO Notation**
   ```
   Instruction: Perform Named Entity Recognition (NER) on the text and use BIO notation (B-TYPE, I-TYPE, O) to tag each token.
   Text: "Steve Jobs founded Apple in Cupertino."
   Expected Output: "Steve [B-PERSON] Jobs [I-PERSON] founded [O] Apple [B-ORGANIZATION] in [O] Cupertino [B-LOCATION] ."
   ```
```

## Best Practices
1. **Determinism (Temperature and Seed):** Use `temperature=0.0` and set a `seed` (if supported by the API) to obtain more deterministic and reproducible results, which are crucial for data extraction tasks.
2. **Clear Instructions and Role:** Define a clear role for the LLM (e.g., "You are an AI assistant specialized in NER") and use explicit instructions about the task, the output format, and the entity categories to be extracted.
3. **Few-Shot Learning:** Provide input and output examples (Few-Shot Prompting) to demonstrate the expected format and entity types, improving accuracy and adherence to the schema.
4. **Functions/Tools (JSON Schema):** Use the function calling capability (Function Calling) or provide a detailed JSON Schema to force the model to return output in a valid, structured JSON format, ideal for integration into data pipelines.
5. **Domain Prompts:** Adapt prompts to the specific domain (e.g., medical, financial, culinary) to improve accuracy in identifying contextual entities.
6. **Chain-of-Thought (CoT):** Ask the model to "think out loud" or justify its extractions before providing the final output, which can increase accuracy in complex cases.
7. **Prompt Chaining:** Break the NER task into smaller steps (e.g., 1. Identify the entity boundary, 2. Classify the entity) to improve performance.

## Use Cases
1. **Document Data Extraction:** Automate the extraction of key information (party names, dates, amounts) from contracts, invoices, financial reports, and legal documents.
2. **Social Media Analysis:** Identify mentions of brands, products, people, and locations across large volumes of social media text for brand monitoring and sentiment analysis.
3. **Biomedicine and Healthcare:** Extract the names of diseases, medications, symptoms, and procedures from medical records and scientific articles.
4. **News and Journalism:** Summarize and categorize news articles, quickly identifying the main actors (people, organizations) and locations.
5. **E-commerce and Catalogs:** Extract product attributes (brand, model, color, size) from text descriptions to enrich catalogs.

## Pitfalls
1. **Hallucinations and Inaccuracy:** LLMs can "hallucinate" entities or classify them incorrectly, especially in niche domains or with ambiguous prompts.
2. **Format Inconsistency:** Without a strict JSON Schema or Function Calling, the model may fail to return the exact requested output format, hindering downstream processing.
3. **Cost and Latency:** Using LLMs for NER can be more expensive and slower than optimized traditional ML models, especially for large volumes of data.
4. **Context Dependency:** Accuracy can drop if the input text is very long and the contextual information needed for entity classification falls outside the model's context window.
5. **Model Biases:** The model may reflect biases present in its training data, leading to biased or incomplete classifications.

## URL
[https://dswithmac.com/posts/prompt-eng-ner/](https://dswithmac.com/posts/prompt-eng-ner/)
