# Computer Vision Prompts

## Description
Prompt Engineering for Computer Vision (especially Vision Language Models - VLMs) is the practice of tuning the text input (the "prompt") provided to a multimodal model (which accepts images, videos, and text) to guide its output and improve the quality and accuracy of the response. This technique is a lightweight and efficient alternative to full fine-tuning of the model, allowing the user to steer the VLM toward specific tasks such as classification, visual question answering (VQA), object detection, and temporal video analysis. The prompt acts as a context that "unlocks" the model's capabilities for a given domain or task.

## Examples
```
1. **Temporal Localization (Video):** `Prompt: When did the worker drop the box?` (A time interval and a description are expected).
2. **Structured Output (VQA):** `Prompt: Is there a fire truck? Is there a fire? Are there firefighters? Present the answer for each question in JSON format.`
3. **Estimation/VQA (Single Image):** `Prompt: Estimate the stock level of the snack table on a scale of 0 to 100.`
4. **Object Detection with Coordinates (Single Image):** `Prompt: Identify all objects of type 'truck' in the image and provide their bounding box coordinates in the format [x_min, y_min, x_max, y_max].`
5. **OCR Extraction and Structuring (Document Analysis):** `Prompt: You are an invoice processor. Extract the 'Invoice Number', 'Due Date', and 'Total Amount' from the document image. Present the output in a JSON object with the keys 'invoice_number', 'due_date', and 'total_amount'.`
6. **Comparison and Quality Control (Multiple Images):** `Prompt: Compare 'Image A' (Reference Product) with 'Image B' (Inspected Product). Describe the differences and classify Product B as 'Approved' or 'Rejected' based on the presence of visible defects.`
7. **Visual Reasoning and Problem Solving (Single Image):** `Prompt: Describe what is happening in the image, and then suggest the next logical action to solve the problem presented. You are a maintenance agent.`
```

## Best Practices
**Clarity and Specificity:** Clearly define the objective, the output format, and the visual focus. **Role:** Assign a role to the model (e.g., "You are a safety inspector..."). **Structured Output:** Request the output in structured formats (such as JSON) to facilitate processing by downstream tasks. **In-Context Learning:** Provide examples of pairs (image/video + prompt + response) to guide the model. **Prompt Tuning (VP/VPT):** For models that support it, using Prompt Tuning (adjusting "soft prompts" with frozen weights) is more efficient and lightweight than full Fine-Tuning. **Temporal Context:** For videos, use models that support sequential understanding to capture the progression of actions.

## Use Cases
**Single Image Understanding:** Classification, captioning, visual question answering (VQA), basic event detection. **Multiple Image Understanding:** Compare, contrast, and learn from multiple visual inputs (e.g., stock level detection over time). **Video Understanding (Temporal Localization):** Understand actions and trends over time, identifying when and where critical events occur. **OCR Detection:** Extract text from images and documents. **Document Analysis:** Process forms or scanned documents. **Security Monitoring:** Detection of anomalies or rule violations in surveillance videos. **Quality Control:** Compare product images to identify defects.

## Pitfalls
**Vague Prompts:** Instructions that are not specific enough or that do not provide sufficient visual focus. **Overloaded Context:** Including too much background information or multiple tasks in a single prompt, diluting the focus (token dilution). **Contradiction:** Prompt elements that cancel each other out or confuse the model. **Ignoring Structured Output:** Not requesting structured formats (JSON, XML) when the output will be used by other systems. **Lack of Domain-Specific Context:** In specific use cases (e.g., retail, medicine), the lack of relevant context can lead to inaccurate responses.

## URL
[https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/](https://developer.nvidia.com/blog/vision-language-model-prompt-engineering-guide-for-image-and-video-understanding/)
