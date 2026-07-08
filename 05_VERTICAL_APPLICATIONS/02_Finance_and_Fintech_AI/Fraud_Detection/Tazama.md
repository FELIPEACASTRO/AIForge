# Tazama

## Description

**Tazama** is the first open-source software platform for real-time financial monitoring and fraud detection, managed by Linux Foundation Charities and funded by the Gates Foundation. Its unique value proposition lies in offering a **global, scalable, and cost-effective** solution for real-time transaction monitoring, focusing especially on **emerging markets** where fraud is prevalent due to the lack of secure digital tools. The project aims to promote financial inclusion, ensuring that Financial Service Providers (FSPs) can execute transactions securely and quickly, reducing the risk of fraud and scams. The name "Tazama" comes from Swahili and means "to take a look", reflecting its function of monitoring events in real time [1] [2]. The project launched in February 2024, with version 2.0 being the most recent [3].

## Statistics

*   **Launch:** February 2024 (Version 2.0) [3].
*   **Management:** Linux Foundation Charities [1].
*   **Funding:** Gates Foundation [1].
*   **Focus:** Emerging markets, promoting financial inclusion [2].
*   **Key Technologies:** Docker, NATS, ArangoDB [6].
*   **Repository Status (GitHub - Full-Stack-Docker-Tazama):** 11 stars, 8 forks (as of 08/11/2025) [6].
*   **Latest Release:** v2.2.0 (August 14, 2025) [6].

## Features

*   **Real-Time Transaction Monitoring:** Ability to monitor each transaction as it occurs, enabling immediate fraud detection and prevention.
*   **Fraud and Scam Prevention:** State-of-the-art software designed to prevent "infection" by fraud and scams, especially in digital payment systems.
*   **Compliance and AML:** Helps improve regulatory compliance and provides features for anti-money laundering (AML) detection, as demonstrated in successful implementations [4].
*   **Microservices-Based Architecture:** Uses a decoupled, microservices-based architecture, which makes it highly scalable and adaptable to different environments.
*   **Configurable Detection Rules:** Although the source code is open, the private "rule processors" (which contain the detection logic to prevent reverse engineering by fraudsters) are deployed from DockerHub with a generic configuration for public access, or with restricted member access for a full multi-typology configuration [5].
*   **Modern Technologies:** Built with technologies such as Docker, NATS (for real-time messaging) and ArangoDB (for the database) [6].

## Use Cases

*   **Monitoring Digital Financial Transactions:** The main use case is real-time monitoring of transactions in digital payment systems, such as transfers and mobile payments, to identify and block fraudulent activity before it is completed [2].
*   **Anti-Money Laundering (AML) Prevention:** The system is designed to help Financial Service Providers (FSPs) meet AML requirements by detecting suspicious transaction patterns that may indicate money laundering [4].
*   **Support for Financial Inclusion:** By providing a low-cost, open-source fraud-detection solution, Tazama enables FSPs in emerging markets to offer safer digital financial services, encouraging the adoption and inclusion of unbanked populations [2].
*   **Detection of Specific Frauds:** The system can monitor and react to various types of fraud and scams, with the detection logic contained in its rule processors [5].

## Integration

Tazama's integration is facilitated by its Docker-based architecture and communication via NATS (a real-time messaging service). Full-stack deployment for demonstration and testing is done using `docker-compose`.

**Integration Method (Example of Environment Variable Configuration):**

Integration with the Transaction Monitoring Service (TMS) and other components is configured through environment variables, as seen in the `ui.env` file of the demo repository [6]:

```javascript
// Example environment variables for the demo user interface (UI)
NEXT_PUBLIC_URL="http://localhost:3001"
NEXT_PUBLIC_TMS_SERVER_URL="http://localhost:5000" // TMS service URL
NEXT_PUBLIC_CMS_NATS_HOSTING="nats://nats:4222" // Connection to the NATS server
NEXT_PUBLIC_ADMIN_SERVICE_HOSTING="http://localhost:5100" // Admin service URL
NEXT_PUBLIC_ARANGO_DB_HOSTING="http://localhost:18529" // ArangoDB database URL
NEXT_PUBLIC_EVENT_TYPES="['pacs.008.001.10', 'pacs.002.001.12', 'pain.001.001.11', 'pain.013.001.09']" // Monitored event/transaction types
```

**Deployment with Docker Compose:**

To start a demo instance, the `Full-Stack-Docker-Tazama` repository provides scripts (`start.sh` for Unix or `start.bat` for Windows) that use `docker-compose` files to orchestrate the microservices (TMS, Admin, NATS, ArangoDB, etc.) [6].

**Event Integration:**

Transaction events are sent to the Tazama system, which processes them in real time using NATS for communication between the microservices. The monitored event types include financial message standards such as `pacs.008.001.10` and `pain.001.001.11` [6].

## URL

https://www.tazama.org/
