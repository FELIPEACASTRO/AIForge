# Test Generation Prompts

## Description
The **Test Generation Prompts** technique consists of crafting structured and detailed instructions for a Large Language Model (LLM) to generate software testing artifacts. This includes test cases, automation scripts, test plans, test data, and even bug reports. The main goal is to leverage the LLM's ability to understand complex requirements and transform them into actionable testing artifacts, increasing the coverage, efficiency, and speed of the QA (Quality Assurance) lifecycle. This technique is fundamental in modern software engineering, especially in the context of agile and DevOps methodologies, where speed and quality are crucial. Effective use of these prompts requires including context, the desired output format, and a clear specification of the type of test and the acceptance criteria.

## Examples
```
**1. Functional Test Case Generation (Markdown Table):**
\`\`\`
Act as a Senior QA Engineer. Generate 10 functional test cases for the "User Login" feature based on the following requirement: "The user must be able to log in with a valid email and a password of 8 to 16 characters. Failed login attempts must display a generic error message. After 3 failed attempts, the account must be locked for 5 minutes."
Output Format: Markdown table with columns: ID, Test Title, Preconditions, Steps, Expected Result, Type (Positive/Negative).
\`\`\`

**2. Automation Script Generation (Python/Selenium):**
\`\`\`
Write an automation test script in Python using the Selenium library to verify the "Add Item to Cart" functionality on an e-commerce site. The script must: 1. Navigate to the product URL. 2. Click the "Add to Cart" button. 3. Verify that the number of items in the cart icon is updated to 1.
Product URL: [Product URL]
\`\`\`

**3. Test Data Generation (JSON):**
\`\`\`
Generate 5 sets of test data in JSON format to test the new user registration API. Include 2 success cases (valid data) and 3 failure cases (e.g., invalid email, password too short, missing required field).
Expected JSON structure: {"name": "string", "email": "string", "password": "string"}.
\`\`\`

**4. Security Test Generation (OWASP Top 10):**
\`\`\`
Based on the following code snippet (or functionality description): [Code Snippet/Description], identify and generate 3 security test scenarios that address the OWASP Top 10 vulnerabilities (e.g., SQL Injection, XSS). For each scenario, provide the attack vector and the expected result.
\`\`\`

**5. Performance Test Generation (JMeter Plan):**
\`\`\`
Create a load test plan for the "Product Search" functionality. The test must simulate 500 concurrent users for 10 minutes. The expected average response time is less than 500ms. Provide the steps to configure this test in Apache JMeter, including the Thread Group and the HTTP Request Sampler.
\`\`\`

**6. Usability Test Scenario Generation (Nielsen Heuristics):**
\`\`\`
Analyze the following user interface (describe the interface or provide a link) and generate 5 usability test scenarios based on Nielsen's Heuristics (e.g., Visibility of System Status, Match Between System and the Real World).
Interface: [Interface Description]
\`\`\`
```

## Best Practices
**1. Provide Complete Context (Contextualization):** Include as much detail as possible about the system, module, functionality, and test environment. Use user documentation, requirements, or code snippets as input. **2. Define the Output Format (Structure):** Specify the exact format you expect (e.g., Markdown table, JSON, Gherkin format, or a specific code script such as Python/Selenium). **3. Specify the Type of Test (Intent):** Be explicit about the desired type of test (e.g., functional, unit, integration, security, performance, usability). **4. Include Constraints and Acceptance Criteria:** Mention any constraints (e.g., "only happy path tests", "cover all validation error cases") and the acceptance criteria for test success. **5. Iterate and Refine (Continuous Refinement):** Use the LLM output as a starting point. Refine the prompt based on the initial results to cover gaps or correct inaccuracies. **6. Use the Persona (Role-Playing):** Ask the LLM to assume the role of a "Senior QA Engineer" or "Security Specialist" to obtain more focused, high-quality results.

## Use Cases
**1. Accelerating Test Case Creation:** Rapid generation of a large volume of test cases from user requirements (User Stories) or functional specifications. **2. Automation Script Creation:** Generation of initial code (e.g., Python, Java, JavaScript) for unit, integration, or UI (User Interface) tests using frameworks such as Selenium, Cypress, or Playwright. **3. Test Data Generation:** Creation of synthetic data sets, valid and invalid, to test APIs and forms, ensuring coverage of different input scenarios. **4. Coverage Gap Identification:** Analysis of an existing set of tests and requirements to suggest additional test scenarios that increase coverage and reduce risk. **5. Test Plan and Strategy Development:** Generation of structured test plans, including scope, resources, schedules, and types of tests to be executed. **6. Specific Test Generation (Security and Performance):** Creation of test scenarios focused on security (e.g., injection, XSS) or performance (e.g., load and stress tests).

## Pitfalls
**1. Over-Reliance on LLM Output:** Assuming that the generated tests are perfect or complete. The LLM may generate syntactically correct but semantically incorrect or incomplete tests. **2. Lack of Specific Context:** Using vague or generic prompts. This leads to superficial test cases that do not cover business rules or system specifics. **3. Ignoring Edge and Negative Cases:** Focusing only on "happy paths". It is crucial to explicitly request negative, exception, and edge case tests. **4. Not Specifying the Format:** Receiving output in an inconsistent format or one that is difficult to integrate with QA tools (e.g., free-flowing text instead of JSON or Gherkin). **5. Requirement Hallucinations:** The LLM may "hallucinate" requirements or functionalities that do not exist, generating irrelevant tests. Always validate the generated tests against the actual documentation. **6. Not Including Acceptance Criteria:** The absence of clear expected results in the prompt can lead to ambiguous or non-verifiable tests.

## URL
[https://www.practitest.com/resource-center/blog/chatgpt-prompts-for-software-testing/](https://www.practitest.com/resource-center/blog/chatgpt-prompts-for-software-testing/)
