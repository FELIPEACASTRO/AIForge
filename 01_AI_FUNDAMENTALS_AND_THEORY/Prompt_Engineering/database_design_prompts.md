# Database Design Prompts

## Description
**Database Design Prompts** are Prompt Engineering techniques that use Large Language Models (LLMs) to assist or automate the process of creating and optimizing database schemas. This category of prompts focuses on providing the AI with business requirements, entities, and desired relationships, requesting in return the generation of DDL (Data Definition Language) code, Entity-Relationship Diagrams (ERD), or architecture recommendations.

The effectiveness of these prompts lies in the AI's ability to simulate the reasoning of a data architect, applying normalization principles, indexing strategies, and scalability and security considerations. They are particularly useful for accelerating the initial design phase, validating conceptual models, and exploring different schema approaches (relational, NoSQL, graph) based on specific use cases. The recent trend (2023-2025) shows an evolution from simple prompts to complex requests that integrate compliance requirements (GDPR, LGPD) and microservices architectures.

## Examples
```
**Example 1: Complete Relational Schema Design (3NF)**
```
Act as a Senior Database Architect. Design a relational database schema in Third Normal Form (3NF) for an e-commerce platform that sells digital and physical products. The system must manage: Customers, Orders, Order Items, Products, Categories, and Reviews. Generate the SQL DDL code for PostgreSQL, including primary keys, foreign keys, and NOT NULL constraints.
```

**Example 2: Performance Optimization (Indexing)**
```
Given the following table schema [INSERT TABLE DDL CODE HERE], and knowing that the most frequent queries involve filtering by 'order_status' and ordering by 'created_at', suggest an optimized indexing strategy. Include the justification for the index type (B-tree, Hash, etc.) and the SQL code to create the indexes.
```

**Example 3: NoSQL Design for Log Data**
```
Design a NoSQL data model (MongoDB) to store access logs from a high-traffic web application. Each log must include: user_id, timestamp, endpoint_accessed, duration_ms, and error_data (if any). The focus is on a high write rate and fast retrieval of logs by 'user_id' and 'timestamp'. Provide the JSON of an example document and the structure of the collection.
```

**Example 4: Data Modeling for Microservices**
```
We are migrating from a monolith to a microservices architecture. The 'Inventory' microservice is responsible for managing product stock. Design the database schema (MySQL) for this microservice, ensuring that it is fully autonomous. The schema must support stock control, warehouse location, and stock reservations. Generate the ER diagram using Mermaid syntax.
```

**Example 5: Inclusion of Compliance Requirements (GDPR/LGPD)**
```
Design the 'Customers' table for a SaaS database that operates in the European Union and Brazil. The design must adhere to the Privacy by Design principles of GDPR/LGPD. Specify which fields must be encrypted (e.g., 'ssn', 'full_name'), how to manage 'consent', and the strategy for the 'right to be forgotten' (anonymization/deletion).
```

**Example 6: Relationship Refinement (Many-to-Many)**
```
I have the entities 'Authors' and 'Books' with a many-to-many relationship. Create the join table 'Authors_Books' and add an extra field called 'author_role' (e.g., 'Lead', 'Co-author'). Generate the DDL code for this join table in SQL Server.
```
```

## Best Practices
**1. Detailed Contextualization:** Always provide as much detail as possible about the project, including the type of application (e-commerce, SaaS, IoT), the expected data volume (small, medium, terabytes), and the primary focus (OLTP, OLAP, Hybrid).
**2. Output Specification:** Explicitly request the desired output format (SQL DDL, ER Diagram in Mermaid/PlantUML, JSON, Markdown).
**3. Constraints and Requirements:** Include crucial non-functional requirements, such as normalization level (3NF, denormalized), security requirements (encryption of sensitive fields), and scalability (sharding, replication).
**4. Iteration and Refinement:** Use subsequent prompts to refine the initial design. For example, "Refine the schema for the 'Orders' table by adding a composite index on 'status' and 'order_date'".
**5. Role Definition:** Begin the prompt by defining the AI's role, such as "Act as a Senior Database Architect with 15 years of experience in distributed systems".

## Use Cases
**1. Rapid Prototyping (MVP):** Quickly generate the initial schema of a database for a Minimum Viable Product (MVP), allowing developers to start coding immediately.
**2. Architecture Validation:** Validate an existing conceptual data model by asking the AI to identify normalization flaws, performance bottlenecks, or scalability issues.
**3. DBMS Migration:** Request the conversion of a schema from one DBMS to another (e.g., from Oracle to PostgreSQL), including the adaptation of data types and DDL syntax.
**4. Automatic Documentation:** Generate ERD diagrams (using syntax such as Mermaid or PlantUML) and detailed schema documentation from a high-level description.
**5. Query Optimization:** Receive suggestions for indexes, table partitioning, and schema optimizations to improve the performance of slow queries in existing databases.
**6. Compliance and Security:** Integrate security and compliance requirements (e.g., PCI DSS, HIPAA, LGPD) directly into the schema design, specifying fields for encryption or anonymization.

## Pitfalls
**1. Lack of Context:** Requesting a database design without specifying the DBMS (PostgreSQL, MySQL, MongoDB), the data volume, or the type of workload (OLTP vs. OLAP) leads to a generic and inefficient design.
**2. Over-reliance on Normalization:** The AI may suggest a highly normalized schema (4NF or 5NF), which is academically correct but can introduce unnecessary complexity and slowness in high-performance systems that would benefit from strategic denormalization.
**3. Ignoring Non-Functional Requirements:** Failing to include security, compliance (GDPR/LGPD), or backup/recovery strategies in the initial prompt results in an incomplete design that will require significant rework.
**4. Entity Ambiguity:** Using ambiguous names or failing to clearly define the entities and their attributes (e.g., what exactly is a "Product"?) will cause the AI to create a model that does not reflect the real business logic.
**5. Failure to Iterate:** Treating the AI's first result as the final design. Database design is an iterative process; the initial prompt should be followed by refinement and validation prompts.

## URL
[https://clickup.com/p/ai-prompts/database-design](https://clickup.com/p/ai-prompts/database-design)
