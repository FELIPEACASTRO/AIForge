# Anomaly Detection Prompts

## Description
Anomaly Detection Prompts use Large Language Models (LLMs) to identify patterns, events, or data points that deviate significantly from normal behavior in a dataset. The technique relies on the LLM's capacity for contextual reasoning and pre-trained knowledge to distinguish what is "normal" from what is "abnormal". The prompt is crucial, as it must convert non-textual data (such as logs, time series, or tabular data) into a format the LLM can process, in addition to providing context, examples (few-shot), and reasoning constraints (Chain-of-Thought) to optimize detection accuracy. The approaches include direct detection, generation of synthetic anomalous data for data augmentation, and generation of natural-language explanations for detected anomalies.

## Examples
```
**1. Anomaly Detection in Server Logs (CoT):**
```
**Instruction:** You are a security analyst. Analyze the log sequence below and determine whether there is an anomaly.
**Logs:**
[2025-11-08 10:00:01] INFO User 'admin' logged in from IP 192.168.1.10
[2025-11-08 10:00:05] DEBUG Database query successful
[2025-11-08 10:00:08] INFO User 'admin' accessed /api/v1/data
[2025-11-08 10:00:10] ERROR Failed to connect to external service: Timeout
[2025-11-08 10:00:11] INFO User 'admin' logged in from IP 203.0.113.55 (New login in 10 seconds)
**Reasoning Steps (CoT):**
1. The first login by 'admin' occurred at 10:00:01.
2. The second login by 'admin' occurred at 10:00:11, just 10 seconds later, from a completely different IP (203.0.113.55).
3. Rapid logins from distinct IPs for the same user are an unusual pattern and suggest session hijacking or unauthorized simultaneous access.
**Anomaly Detected:** Yes/No
**Explanation:** [Generate the explanation based on the reasoning]
```

**2. Anomaly Detection in Time Series (Converted Tabular Data):**
```
**Instruction:** Analyze the CPU usage data (in %) over the last 10 hours. Normal behavior is between 20% and 70%. Identify the most anomalous data point.
**Data:**
Hour: 1, CPU: 35%
Hour: 2, CPU: 42%
...
Hour: 8, CPU: 68%
Hour: 9, CPU: 95%
Hour: 10, CPU: 55%
**Anomaly:** [Anomalous data point]
**Justification:** [Explain why the point is anomalous relative to the normal threshold and the context]
```

**3. Generation of Synthetic Anomalous Data (Data Augmentation):**
```
**Instruction:** You are a security log generator. Create 3 examples of system logs that represent a "Brute Force Attempt" on a web server, keeping the standard log format.
**Standard Format:** [TIMESTAMP] [LEVEL] [SOURCE] [MESSAGE]
**Synthetic Examples:** [Generate 3 logs that simulate the anomaly]
```

**4. Anomaly Detection in Text (Document Review):**
```
**Instruction:** Analyze the following paragraph from a financial report. The expected tone is formal and optimistic. Identify and justify any sentence that presents an anomaly of tone or content.
**Paragraph:** "The quarter's growth exceeded expectations, driven by strategic innovations. However, the leadership team is secretly planning to sell the company to a competitor at a below-market price, which is an imminent disaster."
**Anomaly:** [Anomalous sentence]
**Type of Anomaly:** [E.g.: Factual, Tonal]
**Justification:** [Explain the anomaly]
```

**5. Anomaly Detection in Tabular Data (Zero-Shot):**
```
**Instruction:** Given the table of customer transactions, identify the 'Transaction ID' that represents a value anomaly.
**Table (CSV):**
Transaction ID, Customer, Amount, Location
T001, João, 50.00, SP
T002, Maria, 120.50, RJ
T003, Pedro, 987500.00, MG
T004, Ana, 75.20, SP
**Anomaly:** [Transaction ID]
**Reason:** [Explain the deviation from the pattern]
```
```

## Best Practices
**1. Detailed Prompt Structuring:** Include the task description, the definition of "normal" and "anomaly", the desired output format (e.g., JSON), and the data context (e.g., log type, time series frequency).
**2. Few-Shot Learning:** Provide labeled examples of normal and anomalous data (few-shot examples) to guide the LLM, especially in niche domains where pre-trained knowledge may be insufficient.
**3. Chain-of-Thought (CoT):** Ask the LLM to justify its reasoning before providing the final verdict. This increases interpretability and accuracy by forcing the model to follow a logical process.
**4. Data Conversion:** For non-textual data (time series, tabular), develop a robust pipeline to convert the data into a comprehensible textual format without information loss (e.g., tokenization, statistical description).
**5. Human Validation:** Use the LLM to generate natural-language explanations for the anomalies, facilitating validation and action by human analysts.

## Use Cases
**1. Systems and Security Monitoring (Logs):** Analysis of server, network, or application logs to identify unusual events, such as intrusion attempts, system failures, or error spikes.
**2. Financial Fraud Detection:** Analysis of banking or credit card transactions to identify spending patterns that deviate from the user's normal profile.
**3. Predictive Maintenance (Time Series):** Monitoring of machine sensor data (temperature, vibration, pressure) to detect deviations that signal imminent equipment failure.
**4. Data Quality Control:** Identification of outliers, entry errors, or inconsistent records in large tabular databases.
**5. Healthcare Analysis (Medical Records):** Identification of unusual patterns in electronic health records that may indicate a rare condition or a diagnostic error.

## Pitfalls
**1. Excessive Reliance on Pre-trained Knowledge:** LLMs may fail to detect anomalies in niche domains or in data with very specific patterns that were not present in their training.
**2. Information Loss in Conversion:** Converting non-textual data (time series, images) to text can lead to the loss of crucial details, resulting in false negatives or positives.
**3. Cost and Latency:** Real-time anomaly detection may be infeasible due to the high computational cost and latency of LLM inference, especially for large volumes of data.
**4. Hallucinations:** The LLM may "hallucinate" anomalies or explanations, producing results that seem plausible but are factually incorrect.
**5. Difficulty in Defining "Normal":** A vague or ambiguous definition of "normal behavior" in the prompt can lead to inconsistent results. The prompt should be as specific as possible.

## URL
[https://towardsdatascience.com/boosting-your-anomaly-detection-with-llms/](https://towardsdatascience.com/boosting-your-anomaly-detection-with-llms/)
