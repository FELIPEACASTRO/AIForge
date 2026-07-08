# Data Transformation Prompts

## Description
Data Transformation Prompts are Prompt Engineering instructions designed to guide Large Language Models (LLMs) in the conversion, cleaning, normalization, and restructuring of data from one format or state to another. This technique is fundamental for data preprocessing tasks, where the input (often unstructured, inconsistent, or in a specific format) needs to be converted into a structured and usable output (such as JSON, CSV, SQL, or a specific schema format). Its effectiveness lies in clearly defining the input format, the desired output format, and the manipulation or cleaning rules to be applied. It is widely used in Data Engineering and Data Analysis workflows to automate repetitive tasks and ensure data quality and consistency.

## Examples
```
**1. Format Conversion (CSV to JSON):**
```
Act as a data converter. Your task is to convert the CSV text provided below into an array of JSON objects. Use the CSV headers as JSON keys.

Input CSV:
Name,Age,City
Alice,30,New York
Bob,25,London
Charlie,35,Paris
```

**2. Data Normalization (Address Standardization):**
```
You are a data cleaning agent. Standardize the 'Address' column in the provided list to the format 'Street [Name], No. [Number], [City], [State/Country]'. Correct common abbreviations and typos.

Input Data:
- R. das Flores, 123, SP
- Av. Paulista 456, São Paulo
- 789 Oak St, NY, USA
```

**3. Extraction and Restructuring (Unstructured Text to Markdown Table):**
```
Extract the following information from the text below: Product Name, Price, and Availability. Present the result in a Markdown table.

Input Text:
The new Smartphone X, released in 2024, is available for R$ 4,500.00. Stock is limited. The Y Headphones cost R$ 500 and are out of stock.
```

**4. SQL Code Generation from Requirements:**
```
Based on the following database schema (Table: Orders, Columns: order_id, customer_id, amount, order_date), write a SQL query that returns the 'customer_id' and the total 'amount' of orders placed in the last month.
```

**5. Text Cleaning (Removal of Special Characters and Duplicates):**
```
Clean the following list of customer names. Remove any non-alphanumeric characters (except spaces) and eliminate duplicate names. Return the cleaned list, one name per line.

Customer List:
João Silva!
Maria Souza
João Silva!
Pedro_Alves
Maria Souza
```

**6. Unit Transformation:**
```
Convert all temperature values in the following list from Celsius to Fahrenheit. Return only the new values.

Temperatures in Celsius:
10, 25, 0, 37.5
```

**7. Data Filtering and Aggregation:**
```
Analyze the list of transactions and filter only the transactions with 'status' = 'completed'. Then, calculate the total sum of the 'amount' of those transactions.

Transactions (JSON Format):
[{"id": 1, "status": "pending", "amount": 100}, {"id": 2, "status": "completed", "amount": 250}, {"id": 3, "status": "completed", "amount": 150}]
```
```

## Best Practices
**1. Clear Format Definition:** Always specify the desired output format (e.g., "Return the result strictly in JSON format", "Convert to CSV with commas as the delimiter"). Use output format notation (such as JSON Schema) when possible. **2. Provide Examples (Few-Shot):** For complex or ambiguous transformations, include 1-2 examples of input/output pairs to demonstrate the expected transformation pattern. **3. Explicit Cleaning Instructions:** When cleaning data, explicitly list the cleaning rules (e.g., "Remove duplicates", "Standardize dates to YYYY-MM-DD", "Replace null values with 'N/A'"). **4. Batch Processing (Chunking):** For large volumes of data, split the input into smaller parts and apply the transformation prompt to each part, combining the results afterward. This avoids overflowing the LLM's context limit. **5. Validation and Verification:** Ask the LLM to include a validation step or a summary of the transformations performed, or use external tools to validate the output format (e.g., a JSON validator).

## Use Cases
**1. Data Engineering (ETL/ELT):** Automate the conversion of raw data from logs or legacy systems (e.g., XML, plain text) into structured formats (e.g., JSON, Parquet) ready for ingestion into data warehouses. **2. Data Cleaning and Preprocessing:** Normalize, standardize, and clean datasets for analysis, correcting inconsistencies, removing duplicates, and handling missing values. **3. Code Generation:** Create transformation scripts (Python, SQL, R) from natural language descriptions, accelerating the development of data pipelines. **4. Systems Integration:** Convert message formats between different APIs or services (e.g., from an API response format to an internal database schema). **5. Sentiment Analysis and Classification:** Transform unstructured text (e.g., customer reviews) into categorical or numerical data (e.g., sentiment score, product category) for statistical analysis.

## Pitfalls
**1. Format Ambiguity:** Failing to clearly specify the output format can lead the LLM to return unstructured text or an invalid JSON/CSV format. **2. Context Window:** Attempting to transform large datasets all at once may exceed the LLM's token limit, resulting in truncation or transformation failure. **3. Data Typing Errors:** The LLM may misinterpret the data type (e.g., treating a number as a string) if typing instructions are not explicit. **4. Overly Complex Transformation:** Asking the LLM to perform multiple complex transformations (cleaning, conversion, aggregation) in a single step increases the chance of errors. It is better to use **Prompt Chaining**. **5. Hallucinations in Cleaning:** Instead of correcting data errors, the LLM may "hallucinate" nonexistent data or make incorrect assumptions if the cleaning rules are not strict.

## URL
[https://stratpilot.ai/10-powerful-ai-prompts-for-data-transformation/](https://stratpilot.ai/10-powerful-ai-prompts-for-data-transformation/)
