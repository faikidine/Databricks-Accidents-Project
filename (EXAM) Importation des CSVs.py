# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Importation des fichiers et transformation en tables

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Caractéristiques

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/carac.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ";"

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "CARACTERISTIQUES"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC
# MAGIC select * from `CARACTERISTIQUES`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "CARACTERISTIQUES"

df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Lieux

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/lieux.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ";"

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "LIEUX"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

permanent_table_name = "LIEUX"

df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Victoires

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/vict.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ";"

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "VICTIMES"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

permanent_table_name = "VICTIMES"

df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Vehicules

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/veh.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ";"

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "VEHICULE"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

permanent_table_name = "VEHICULE"

df.write.format("parquet").saveAsTable(permanent_table_name)
