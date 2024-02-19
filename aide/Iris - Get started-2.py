# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Lire les données

# COMMAND ----------

# Importer les bibliothèques nécessaires
from sklearn.datasets import load_iris
import pandas as pd

# Charger le jeu de données Iris
iris_data = load_iris()

# Les données sont stockées dans iris_data.data (les caractéristiques) et iris_data.target (les étiquettes/classes)

# Créer un DataFrame pandas pour visualiser les données
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['target'] = iris_data.target

# Afficher les 5 premières lignes du DataFrame
print(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Train / Test

# COMMAND ----------

# Importer les bibliothèques nécessaires
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Charger le jeu de données Iris
iris_data = load_iris()

# Diviser les données en jeu d'entraînement et jeu de test
X_train, X_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.30, random_state=42)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # Fit KNN model

# COMMAND ----------

# Créer un modèle K-NN avec k=3
knn_model = KNeighborsClassifier(n_neighbors=5)

# Entraîner le modèle sur le jeu d'entraînement
knn_model.fit(X_train, y_train)

# Prédire les étiquettes/classes sur le jeu de test
y_pred = knn_model.predict(X_test)

# Calculer l'exactitude du modèle sur le jeu de test
accuracy = accuracy_score(y_test, y_pred)
print("Exactitude du modèle K-NN : {:.2f}".format(accuracy))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Sauvegarder un modèle

# COMMAND ----------

import mlflow
import mlflow.sklearn
# Créer un modèle K-NN avec k=3
knn_model = KNeighborsClassifier(n_neighbors=3)

# Entraîner le modèle sur le jeu d'entraînement
knn_model.fit(X_train, y_train)

# Sauvegarder le modèle avec MLflow
mlflow.sklearn.log_model(knn_model, "modelName")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Inscrire son modèle avant deploiement

# COMMAND ----------

# Register the model with a specific name and version
model_uri = "runs:/xxx/modelName"
registered_model_name = "MyRegisteredModel"
model_version = "1.0"

model_details = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name
)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Charger son modèle

# COMMAND ----------

import mlflow
logged_model = 'runs:/xxx/modelName'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(X_test)
