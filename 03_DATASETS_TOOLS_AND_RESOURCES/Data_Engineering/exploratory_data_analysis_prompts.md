# Exploratory Data Analysis (EDA) Prompts

## Description
Exploratory Data Analysis (EDA) Prompts are structured instructions provided to Large Language Models (LLMs) to automate, guide, and accelerate the EDA process. EDA is a fundamental step in data science that aims to summarize the main characteristics of a dataset, often using visual methods. By using prompts, the analyst can delegate tasks such as data cleaning, computing descriptive statistics, identifying *outliers*, detecting missing values, and suggesting visualizations. Effective use of EDA prompts turns the LLM into an interactive data assistant, allowing the analyst to focus on interpreting the *insights* rather than on repetitive coding. The key is to provide clear context, define the data structure, and request actionable output (code, tables, or executive summaries).

## Examples
```
**1. Complete Descriptive Statistical Analysis**
```
Act as a Senior Data Scientist. Analyze the CSV file 'vendas_mensais.csv'. Generate a Markdown table that includes: count of non-null values, mean, standard deviation, minimum, maximum, Q1, Q2 (median), and Q3 for all numeric columns. For categorical columns, list the count of unique values and the mode.
```
**2. Outlier and Missing Value Identification**
```
Based on the dataset 'dados_clientes.csv', identify all columns with more than 5% missing values. For the 'Renda_Anual' column, use the IQR method to detect and list the top 10 outliers. Suggest the best imputation strategy for the missing values in the 'Idade' column.
```
**3. Visualization Suggestions for Variable Relationships**
```
My goal is to understand the relationship between 'Tempo_de_Serviço' (numeric) and 'Taxa_de_Churn' (binary) in 'dataset_telecom.csv'. Suggest the 3 most informative chart types to visualize this relationship. For each chart, provide the Python code (using Matplotlib or Seaborn) and the expected interpretation.
```
**4. Distribution and Normality Analysis**
```
Focus on the 'Preço_do_Imóvel' column of 'dataset_imoveis.csv'. Describe the shape of the distribution (symmetric, left/right skewed). Compute the kurtosis and the skewness. Based on these results, what can you infer about the normality of the data?
```
**5. Subgroup Segmentation and Comparison**
```
Using 'dataset_marketing.csv', compare the 'Taxa_de_Conversão' and 'Custo_por_Aquisição' metrics across the subgroups defined by the 'Canal_de_Marketing' column (Email, Social, Search). Present the results in a comparative table and highlight the channel with the best ROI.
```
```

## Best Practices
**1. Provide Context and Structure:** Always define the **Persona** (e.g., "You are a Senior Data Scientist"), the **Objective** (e.g., "Find anomalies"), and the **Output Format** (e.g., "Markdown table with 3 columns: Variable, Statistic, Value"). **2. Insert Dataset Metadata:** Mention the dataset name, the number of rows/columns, and, crucially, list the relevant columns and their data types (categorical, numeric, temporal). **3. Iteration and Refinement:** Start with broad prompts (basic statistical analysis) and refine with more specific prompts (investigating *outliers* in a specific column) based on previous outputs (Chain-of-Thought for EDA). **4. Request Code and Explanation:** Explicitly ask for the code (Python/R) used for the analysis, along with a step-by-step explanation of the results and their business implications. **5. Handling Sensitive Data:** Never insert confidential data directly into the prompt. Instead, use synthetic data samples or summary statistics, or use AI tools that ensure privacy and local processing of the data.

## Use Cases
**1. Data Cleaning and Preprocessing:** Generating code to standardize formats, handle missing values (imputation), and correct typographical errors in large datasets. **2. Descriptive Statistics Generation:** Automating the computation of central tendency, dispersion, and shape metrics for all variables in a *dataset*. **3. Visualization Creation:** Suggesting and generating code for informative charts (histograms, *box plots*, scatter plots) to understand the distribution and relationships between variables. **4. Anomaly and *Outlier* Identification:** Using prompts to apply statistical methods (such as Z-score or IQR) to flag unusual data points that require investigation. **5. EDA Executive Summary:** Synthesizing the main *insights* from the analysis into an easy-to-understand format for non-technical *stakeholders*, highlighting business implications. **6. *Feature Engineering*:** Suggesting new variables or data transformations that can improve the performance of subsequent *Machine Learning* models.

## Pitfalls
**1. Over-Reliance on the Output:** Assuming that the LLM's code or analysis is 100% correct without validation. The LLM can make statistical or logical errors (*hallucinations*). **2. Vague or Incomplete Prompts:** Failing to provide the necessary data context (column names, data types, analysis objective), resulting in generic or irrelevant outputs. **3. Ignoring Prompt Structure:** Not using a clear structure (Persona, Context, Task, Format), which reduces the quality and consistency of the response. **4. Inserting Confidential Data:** The security and privacy risk of pasting large volumes of sensitive data directly into the LLM interface. **5. Failure to Iterate:** Treating the LLM as a one-off query tool rather than an interactive partner. EDA is an iterative process, and prompts should reflect this, refining questions based on previous findings.

## URL
[https://team-gpt.com/blog/chatgpt-prompts-for-data-analysis](https://team-gpt.com/blog/chatgpt-prompts-for-data-analysis)
