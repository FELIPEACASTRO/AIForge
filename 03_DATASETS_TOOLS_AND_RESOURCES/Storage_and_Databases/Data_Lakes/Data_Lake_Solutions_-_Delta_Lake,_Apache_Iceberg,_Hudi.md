# Data Lake Solutions - Delta Lake, Apache Iceberg, Hudi

## Description

Delta Lake é uma camada de armazenamento de código aberto que traz confiabilidade (transações ACID), escalabilidade e desempenho para data lakes. Sua proposta de valor única é a arquitetura Lakehouse, unificando ETL, data warehouse e Machine Learning em um formato universal. É o formato de tabela original e mais maduro, com forte suporte da Databricks [1]. Apache Iceberg é um formato de tabela de alto desempenho para grandes tabelas analíticas. Sua proposta de valor única é trazer a confiabilidade e a simplicidade das tabelas SQL para o Big Data, resolvendo problemas de particionamento e garantindo a consistência transacional em arquiteturas multi-engine [2]. Apache Hudi é uma plataforma de data lakehouse de código aberto que fornece funcionalidades de banco de dados (como upserts e exclusões) sobre o data lake. Sua proposta de valor única é o foco em cargas de trabalho de ingestão de dados em tempo real e CDC (Change Data Capture), oferecendo recursos avançados de indexação e serviços de tabela gerenciados [3].

## Statistics

**Delta Lake:** 8.4k estrelas no GitHub, 1.9k forks, 4.493 commits, 214 observadores [4]. **Apache Iceberg:** 8.2k estrelas no GitHub, 2.9k forks, 7.675 commits, 186 observadores [5]. **Apache Hudi:** 6.0k estrelas no GitHub, 2.4k forks, 6.799 commits, 1.1k observadores [6]. O Hudi é notável por ter sido pioneiro em muitos recursos de lakehouse, como Merge-on-Read (out/2017) e Consultas Incrementais (mar/2017), que foram adotados posteriormente por outros projetos [7].

## Features

**Delta Lake:** Transações ACID, Time Travel (viagem no tempo), Schema Enforcement e Evolution, Upserts/Deletes (via `MERGE INTO`), Suporte a Streaming e Batch unificado, Z-Ordering (otimização de layout de dados), Suporte a múltiplas engines (Spark, Flink, PrestoDB, Trino) [1]. **Apache Iceberg:** Evolução de Schema segura e transparente, Evolução de Partição, Ocultação de Particionamento (Partitioning Hiding), Suporte a múltiplas engines (Spark, Flink, Trino, Presto, Hive), Time Travel, Rollback, Suporte a Deletion Vectors (para MOR) [2]. **Apache Hudi:** Transações ACID, Suporte a dois tipos de tabela (Copy-on-Write e Merge-on-Read), Indexação em nível de registro (para upserts rápidos), Serviços de Tabela Gerenciados (Compaction, Cleaning, Clustering), CDC (Change Data Capture) de primeira classe, Suporte a múltiplas engines (Spark, Flink, Hive) [3].

## Use Cases

**Delta Lake:** Construção de Lakehouses, Data Warehousing, Ingestão de dados em tempo real (Streaming), Data Science e Machine Learning (ML) com dados consistentes, e governança de dados [1]. **Apache Iceberg:** Plataformas de pesquisa quantitativa de alto desempenho, gerenciamento de grandes conjuntos de dados analíticos com requisitos rigorosos de consistência, e ambientes multi-engine onde diferentes ferramentas precisam acessar os mesmos dados de forma segura [2]. **Apache Hudi:** Ingestão de dados de baixa latência e CDC (Change Data Capture) de bancos de dados (ex: Uber, Robinhood, Zendesk), processamento de dados fora de ordem, e construção de pipelines de dados que exigem atualizações e exclusões eficientes em nível de registro [3].

## Integration

**Delta Lake (PySpark):**
```python
from delta.tables import *
from pyspark.sql.functions import *

# Configuração do Spark com Delta Lake
builder = pyspark.sql.SparkSession.builder.appName("DeltaLakeApp") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Exemplo de MERGE (Upsert)
deltaTable = DeltaTable.forPath(spark, "/tmp/delta-table")
newData = spark.range(0, 20)

deltaTable.alias("oldData") \
  .merge(
    newData.alias("newData"),
    "oldData.id = newData.id") \
  .whenMatchedUpdate(set = { "id": col("newData.id") }) \
  .whenNotMatchedInsert(values = { "id": col("newData.id") }) \
  .execute()
```
**Apache Iceberg (Spark SQL):**
```sql
-- Configuração do catálogo (exemplo com Hive Metastore)
spark-sql --packages org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.10.0 \
    --conf spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions \
    --conf spark.sql.catalog.spark_catalog=org.apache.iceberg.spark.SparkSessionCatalog \
    --conf spark.sql.catalog.spark_catalog.type=hive

-- Criação de tabela
CREATE TABLE demo.nyc.taxis (
    vendor_id bigint,
    trip_id bigint,
    trip_distance float,
    fare_amount double,
    store_and_fwd_flag string
) PARTITIONED BY (vendor_id);

-- Inserção de dados
INSERT INTO demo.nyc.taxis VALUES (1, 1000371, 1.8, 15.32, 'N');
```
**Apache Hudi (PySpark):**
```python
# Configuração do Spark com Hudi (exemplo de shell)
# spark-shell --packages org.apache.hudi:hudi-spark3.5-bundle_2.12:1.0.2 ...

# Exemplo de Upsert
tableName = "trips_table"
basePath = "file:///tmp/trips_table"

updatesDf.write.format("hudi"). \
  option("hoodie.datasource.write.operation", "upsert"). \
  option("hoodie.datasource.write.partitionpath.field", "city"). \
  option("hoodie.table.name", tableName). \
  mode("append"). \
  save(basePath)
``` [1] [2] [3]

## URL

Delta Lake: https://delta.io/ | Apache Iceberg: https://iceberg.apache.org/ | Apache Hudi: https://hudi.apache.org/