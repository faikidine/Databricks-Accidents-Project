# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Lister les fichiers dans le DBFS

# COMMAND ----------

# MAGIC %fs 
# MAGIC ls dbfs:/

# COMMAND ----------

# MAGIC %fs 
# MAGIC ls dbfs:/FileStore/

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Importer un csv dans le Filestore

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Importer un des 4 fichiers csv sur les accidents sans créer de table

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Convertir un dataframe Spark en pandas

# COMMAND ----------

path = '/FileStore/tables/filename.csv'
dfspark = spark.read.csv(path, sep = ";")
df = dfspark.toPandas()
print(df.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Convertir un dataframe pandas en spark

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession

pandas_df = pd.DataFrame({"name": ["John", "Mary", "Mike"], "age": [25, 30, 35]})

spark = SparkSession.builder.appName("Create DataFrame").getOrCreate()
df = spark.createDataFrame(pandas_df)
df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Sauvegarder dans une table

# COMMAND ----------

df.write.saveAsTable("nom_table")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Importer une table

# COMMAND ----------

# SQL query to select all columns from the table
query = "SELECT * FROM hive_metastore.default.nom_table"

# Execute the query and create a DataFrame
df = spark.sql(query)

# Show the DataFrame
df.show()
