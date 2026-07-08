# Docker & Kubernetes Prompts

## Description
**Docker & Kubernetes Prompts** is a category of Prompt Engineering focused on using Large Language Models (LLMs) to automate, optimize, and troubleshoot container-based and orchestration-based development and production environments. This technique fits within the context of **PromptOps** or **AI-assisted DevOps**, where the AI acts as a copilot for DevOps engineers, SREs, and developers. The main goal is to accelerate the creation of configuration files (such as `Dockerfile` and Kubernetes YAML manifests), diagnose complex failures, and generate automation scripts, transforming natural-language descriptions into actionable infrastructure code. The effectiveness of this technique depends on clarity, specificity, and the provision of detailed technical context in the prompts.

## Examples
```
1.  **Optimized Dockerfile Generation (Multi-Stage):**
    ```
    Create a multi-stage Dockerfile for a Python (Flask) application that uses the 'python:3.11-slim' base image. The build stage must install the dependencies from 'requirements.txt'. The final stage must use 'python:3.11-slim' and copy only the source code and the installed dependencies. Ensure that the final runtime user is non-root and that the pip cache is cleared.
    ```

2.  **Kubernetes Manifest Generation (Deployment and Service):**
    ```
    Generate a Kubernetes YAML manifest that includes a Deployment and a Service. The Deployment must have 3 replicas, use the 'minha-app:v1.2.0' image, and expose port 8080. The Service must be of type LoadBalancer and route traffic to the Deployment. Add a readinessProbe that checks the '/health' endpoint on port 8080.
    ```

3.  **Troubleshooting (CrashLoopBackOff):**
    ```
    I am getting the 'CrashLoopBackOff' error on my Pod. Analyze the container logs (logs attached below) and the Deployment YAML manifest (also attached). Identify the likely cause and suggest the exact fix in the YAML manifest.

    [Container Logs]
    ...
    [YAML Manifest]
    ...
    ```

4.  **Optimizing an Existing Dockerfile:**
    ```
    Analyze the Dockerfile provided below. Suggest 3 optimizations to reduce the final image size and build time, focusing on layer caching and security best practices. Present the optimized Dockerfile.

    [Existing Dockerfile]
    ...
    ```

5.  **Ingress Configuration Creation:**
    ```
    Create a Kubernetes Ingress manifest that routes traffic from the host 'api.meudominio.com' to the Service named 'api-service' on port 80. The Ingress must use TLS with a Secret named 'meu-tls-secret'.
    ```

6.  **Shell Script Generation for K8s:**
    ```
    Write a shell script that checks the status of all Pods in the 'producao' namespace. If any Pod is in a 'CrashLoopBackOff' or 'ImagePullBackOff' state, the script must print the Pod name and its recent logs.
    ```

7.  **HPA Explanation and Generation:**
    ```
    Explain the concept of the Horizontal Pod Autoscaler (HPA) in Kubernetes. Then, generate an HPA manifest for the Deployment named 'web-app-deployment' that maintains average CPU usage at 70%, with a minimum of 2 and a maximum of 10 replicas.
    ```

8.  **YAML Validation and Correction:**
    ```
    Validate the Kubernetes YAML manifest below. Fix any syntax, indentation, or API version errors. Keep the original logic intact and return only the corrected YAML.

    [YAML with Error]
    ...
    ```

9.  **ConfigMap Generation for Environment Variables:**
    ```
    Create a ConfigMap named 'app-config' with the following environment variables: 'LOG_LEVEL'='INFO', 'FEATURE_TOGGLE'='true', and 'API_URL'='http://backend-service'. Then, show how to reference this ConfigMap in a Deployment YAML.
    ```

10. **Dockerfile Security Review:**
    ```
    Review the Dockerfile below to identify and fix security vulnerabilities. The fixes should include removing exposed passwords or keys, ensuring that a non-root user is used, and updating outdated packages.

    [Dockerfile for Review]
    ...
    ```
```

## Best Practices
**Extreme Clarity and Specificity:** Treat the LLM as a "brilliant but unreliable junior engineer." Be extremely detailed about the language version, the base image, the security requirements (e.g., non-root user), and the exact resource type (e.g., Deployment, Service, Ingress).
**Structured and Iterative Communication:** Use iterative prompts. First, ask for the code to be generated. Then, ask for validation and error correction. Finally, ask for optimization (e.g., "Now, optimize this Dockerfile for a smaller final image").
**Inclusion of Context and Constraints:** Always provide relevant context (error logs, existing code, network requirements) and constraints (e.g., "The Service must be of type LoadBalancer", "The HPA must maintain CPU usage at 70%").
**Validation and Security:** Explicitly ask the LLM to validate the generated code (e.g., "Check the YAML syntax and the API version") and to apply security best practices (e.g., "Add a healthcheck and ensure that a non-root user is used").

## Use Cases
**Configuration Automation:** Rapid generation of optimized `Dockerfile`s (multi-stage, lightweight base image) and Kubernetes YAML manifests (Deployment, Service, Ingress, HPA) from natural-language requirements.
**Troubleshooting:** Diagnosis and fix suggestions for common Kubernetes errors (e.g., `CrashLoopBackOff`, `ImagePullBackOff`) based on provided logs and manifests.
**Infrastructure Optimization:** Optimizing Dockerfiles to reduce image size and build time, and suggesting Kubernetes configurations for better scalability and resilience.
**Script and Documentation Generation:** Creating shell scripts for K8s maintenance and automation tasks, and generating technical documentation from existing configurations.
**Security Review:** Analyzing Dockerfiles and Kubernetes manifests to identify and fix security vulnerabilities (e.g., use of the root user, exposure of secrets).

## Pitfalls
**Blind Trust in Generated Code:** The LLM may generate YAMLs with outdated API versions, Dockerfiles without a `healthcheck`, or insecure configurations. Manual validation or validation by specific AI tools (such as K8sGPT) is crucial.
**Exposure of Sensitive Data:** Inserting production logs, secrets, or proprietary data into prompts for public AI models can violate security policies and confidentiality agreements. Using locally hosted models or ones with privacy guarantees is recommended.
**Vague Prompts:** Generic prompts result in unoptimized, insecure code that does not meet the specific requirements of the production environment. The lack of context (e.g., base image, language version) leads to low-quality results.
**Ignoring Iteration:** Expecting a perfect result on the first prompt. Prompt engineering for DevOps is an iterative process of refinement and correction.

## URL
[https://medium.com/@osomudeyazudonu/10-ai-prompts-every-devops-engineer-should-use-to-work-10-faster-3474ac59ffc1](https://medium.com/@osomudeyazudonu/10-ai-prompts-every-devops-engineer-should-use-to-work-10-faster-3474ac59ffc1)
