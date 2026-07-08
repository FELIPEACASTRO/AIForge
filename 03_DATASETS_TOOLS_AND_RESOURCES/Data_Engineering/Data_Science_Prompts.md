# Data Science Prompts

## Description
Data Science Prompts are structured, detailed instructions provided to Large Language Models (LLMs) to assist across every stage of the Data Science and Machine Learning (ML) lifecycle. This ranges from data cleaning and preprocessing, Exploratory Data Analysis (EDA), Feature Engineering, model building and debugging, through to communicating and interpreting results for stakeholders. The effectiveness of these prompts lies in their ability to turn complex, time-consuming tasks into clear commands, leveraging the LLM as a coding, analysis, and communication assistant. The technique aims to increase productivity, automate repetitive tasks, and ensure that analyses and model results are translated into actionable business insights.

## Examples
```
**1. Data Cleaning and Preprocessing (Intermediate)**
```
## System
You are an experienced data cleaning assistant.

## User
Here is the summary of my Pandas DataFrame (include the output of df.info() and df.describe()).

## Task
1. Identify columns with more than 20% missing values.
2. Suggest the best imputation strategy for the numerical and categorical columns.
3. Provide the Python code (using pandas or scikit-learn) to perform the cleaning and imputation, explaining each step.
```

**2. Exploratory Data Analysis (EDA) (Intermediate)**
```
## System
You are a data analysis storyteller focused on business trends.

## User
I have a sales dataset with the columns: 'data', 'id_produto', 'regiao', 'unidades_vendidas', 'preco'.

## Task
Create an EDA checklist to examine seasonality, outliers, and sales trends. Include the Python code for:
1. A line chart for sales over time.
2. A boxplot of 'unidades_vendidas' by 'regiao'.
3. Interpret the results of each visualization in terms of business impact.
```

**3. Feature Engineering (Advanced)**
```
## System
You are a Machine Learning Feature Engineer.

## User
I am building a regression model to predict real estate prices. The available columns are: 'area_quadrada', 'quartos', 'banheiros', 'ano_construcao', 'bairro'.

## Task
1. Suggest 5 derived features that could improve the model's performance (e.g., 'idade_imovel').
2. Write the Python code using pandas to create these 5 features.
3. Generate a Scikit-learn Transformer class to encapsulate these transformations.
```

**4. Model Interpretation (Executive Communication)**
```
## System
You are a senior data storyteller specializing in executive communication.

## User
Here are the SHAP values (feature, impact): [('renda', 0.45), ('idade', 0.20), ('historico_credito', 0.15), ('divida', 0.10)].

## Task
1. Rank the top 3 risk factors by absolute impact.
2. Write a ~120-word summary for the Board of Directors, explaining what increases and what reduces risk.
3. Suggest two concrete mitigation actions.

## Constraints & Style
- Audience: Board level, non-technical.
- Tone: Confident and insight-focused.
- Format: Bullet-point list in Markdown.
```

**5. Model Debugging (Overfitting)**
```
## System
You are a Machine Learning expert.

## User
My sklearn RandomForestClassifier model shows high training accuracy (98%) and low validation accuracy (75%).

## Task
1. List 3 likely reasons for the overfitting.
2. For each reason, provide a suggested fix and the corresponding Python code (e.g., hyperparameter tuning, cross-validation).
3. Explain the concept of "bias-variance" in simple terms.
```
```

## Best Practices
**1. Be Specific and Structured:** Define the LLM's role (e.g., "You are an experienced Feature Engineer"), provide the context (DataFrame schema, business problem), and use clear delimiters (such as `###` or `##`).
**2. Task Decomposition (Chaining):** For complex workflows (cleaning, EDA, modeling), break the task into modular, chained prompts, where the output of one prompt serves as the input to the next.
**3. Require Both Code and Explanation:** Explicitly ask for the code (Python, SQL, R) and for a line-by-line explanation or a summary of the reasoning behind the solution.
**4. Define the Audience and Tone:** When requesting reports or summaries of results, specify the target audience (e.g., "Non-technical executives", "Junior Data Scientists") and the desired tone (e.g., "Concise and cost-focused", "Instructional and detailed").
**5. Use Few-Shot Learning:** Include examples of the desired input and output to guide the model, especially for formatting or data transformation tasks.

## Use Cases
**1. EDA and Data Cleaning Automation:** Generate code to identify and handle missing values, outliers, and format inconsistencies in large datasets.
**2. Accelerated Feature Engineering:** Create complex derived features (e.g., time-lag variables, high-cardinality encoding) and encapsulate them in reusable classes.
**3. Explaining Model Metrics:** Translate complex metrics (e.g., Confusion Matrix, F1-Score) into terms of financial or operational impact for a business audience.
**4. Data Storytelling:** Transform technical model results (e.g., SHAP values, regression coefficients) into concise, actionable narratives for executive reports.
**5. Synthetic Data Generation:** Create synthetic data samples that mimic the distribution and characteristics of a real dataset for testing and development purposes.
**6. Model Debugging and Optimization:** Diagnose common ML problems (overfitting, underfitting, concept drift) and suggest code solutions to fix them.

## Pitfalls
**1. Overly Technical Language:** Using ML jargon (e.g., "AUC", "ROC", "SHAP") when communicating with non-technical stakeholders. The prompt should require translation into business terms (e.g., "cost of a false positive").
**2. Lack of Context:** Failing to provide the dataset schema, the business problem, or the expected input/output format. The LLM may generate irrelevant code or analyses.
**3. Blind Trust in Code:** Accepting generated code without review. The LLM may make subtle logic errors or use outdated libraries. The prompt should include a "self-assessment" or "code review" step.
**4. 'Black Box' Prompts:** Asking only for the final result without requiring the reasoning (Chain-of-Thought). This makes debugging and understanding the analysis process difficult.
**5. Not Specifying the Output Format:** If the result is used in an automated pipeline, the lack of a format specification (JSON, CSV, Python code) can break the workflow.

## URL
[https://towardsdatascience.com/the-end-to-end-data-scientists-prompt-playbook/](https://towardsdatascience.com/the-end-to-end-data-scientists-prompt-playbook/)
