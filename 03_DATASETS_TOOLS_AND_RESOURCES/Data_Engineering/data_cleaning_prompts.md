# Data Cleaning Prompts

## Description
The **Data Cleaning Prompts** technique uses natural-language commands to instruct Large Language Models (LLMs) to identify, correct, standardize, and remove errors or inconsistencies in datasets. This approach turns manual, repetitive tasks such as filling in missing values, removing duplicates, and standardizing formats into automated, efficient processes. Instead of writing complex scripts or using lengthy formulas, the data analyst or ML engineer can simply describe the desired action (e.g., "Fill missing values in Column D with the column's median"), allowing the LLM to execute the cleaning logic based on its vast knowledge of data patterns and formatting rules [1]. This significantly accelerates the critical data preparation phase, which has historically consumed most of the time in analysis and Machine Learning projects [1].

## Examples
```
1. **Filling Missing Data**: "Fill all missing values in the 'Age' column using the column's median and justify the choice of method."
2. **Removing Duplicates**: "Find and remove duplicate rows in the dataset, considering only the 'Customer ID' and 'Transaction Date' columns, and keep the most recent record."
3. **Date Format Standardization**: "Convert all dates in the 'Start Date' column to ISO 8601 format (YYYY-MM-DD)."
4. **Text Standardization (Case)**: "Standardize all product names in the 'Product Name' column to Title Case (First Letter of Each Word Capitalized)."
5. **Typo Correction**: "Identify and correct common spelling errors and variations of city names in the 'Location' column, using a list of Brazilian cities as a reference."
6. **Outlier Detection and Flagging**: "Flag all transactions in the 'Sale Value' column that are 3 standard deviations above or below the mean, and list the 5 largest anomalies."
7. **Extraction and Categorization**: "Extract the area code (DDD) from the phone numbers in the 'Phone' column and create a new column called 'DDD' with this information."
8. **Conditional Cleaning**: "Remove records where the 'Status' column is 'Canceled' AND the 'Cancellation Date' column is empty."
```

## Best Practices
**Be Clear and Specific**: Instead of "Clean my data," use "Find and remove duplicate rows in Column C, keeping the first occurrence" [1]. **Use Action Words**: Prefer direct verbs such as *Convert*, *Clean*, *Find*, *Remove*, *Standardize*, or *Categorize* [1]. **Specify Columns and Formats**: Always define the scope of the action (e.g., "in Column D") and the desired output format (e.g., "to the YYYY-MM-DD format") [1]. **Define Conditions**: Include conditional logic when necessary (e.g., "Remove duplicates only if Column B and Column C are identical") [1]. **Batch Processing (Chunking)**: For large volumes of data, split the dataset into smaller parts (chunks) and process them sequentially or in parallel, since LLMs have context limits [2]. **Post-Cleaning Validation**: Use prompts to validate the result, such as "Verify that all emails in Column F contain '@' and a valid domain" [2].

## Use Cases
**Data Analysis and Business Intelligence (BI)**: Ensuring that the input data for reports and BI dashboards is accurate and consistent, avoiding decisions based on flawed information. **Machine Learning (ML)**: Quickly preparing and preprocessing large volumes of data for model training, standardizing *features* and handling missing values, which is vital for model performance. **System Migration and Data Integration**: Standardizing formats and resolving inconsistencies across different data sources (e.g., old CRM and new ERP) during migration processes. **E-commerce and Product Catalogs**: Correcting typos, standardizing product names and descriptions, and automatically categorizing items to improve customer experience and inventory management. **Financial and Regulatory Sector**: Ensuring data compliance (e.g., KYC - Know Your Customer) through the identification and correction of duplicate or incomplete records [1].

## Pitfalls
**Vagueness and Ambiguity**: Prompts like "Fix the format" or "Remove bad data" are too vague and lead to incorrect or incomplete results [1]. **Ignoring Context Limits**: Attempting to process very large datasets all at once can exceed the LLM's token limit, resulting in truncation or failure [2]. **Over-Reliance on AI**: The AI can introduce new errors (hallucinations) or apply the cleaning logic incorrectly. Human validation and review of the results are crucial [2]. **Not Specifying the Output Format**: Failing to request the result in a structured format (e.g., CSV, JSON, or just the list of corrections) can make it difficult to apply the changes back to the original dataset. **Not Providing Enough Context**: For complex tasks (e.g., filling in missing values), the AI needs context about neighboring columns and the data type to make accurate inferences.

## URL
[https://numerous.ai/blog/ai-prompts-for-data-cleaning](https://numerous.ai/blog/ai-prompts-for-data-cleaning)
