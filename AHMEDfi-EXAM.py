# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # TD - Exam

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Charger les données
# MAGIC J'ai déjà importé les CSVs depuis `Catalog>Add data>DBFS` (voir le notebook `(EXAM)Importation des CSVs`), je les ai transformés en tables dans `default.caracteristiques`, `default.lieux`, ...  
# MAGIC Dans cette partie je vais transformer ces tables en dataframe Spark puis en dataframe Pandas`

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Caractéristiques
# MAGIC Contenu dans le fichier carac.csv

# COMMAND ----------

# requete SQL pour sélectionner toutes les colones d'une table
query = "SELECT * FROM hive_metastore.default.caracteristiques"

# exécution de la requete et enregistrement dans un dfSpark
df = spark.sql(query)

# transformation des dfSpark en Pandas
carac = df.toPandas()

# COMMAND ----------

# afficher un extrait des données avec head
carac.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Lieux
# MAGIC Contenu dans le fichier lieux.csv

# COMMAND ----------

# requete SQL pour sélectionner toutes les colones d'une table
query = "SELECT * FROM hive_metastore.default.lieux"

# exécution de la requete et enregistrement dans un dfSpark
df = spark.sql(query)

# transformation des dfSpark en Pandas
lieux = df.toPandas()

# COMMAND ----------

# afficher un extrait des données avec head
lieux.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Victimes
# MAGIC Contenu dans le fichier vict.csv

# COMMAND ----------

# requete SQL pour sélectionner toutes les colones d'une table
query = "SELECT * FROM hive_metastore.default.victimes"

# exécution de la requete et enregistrement dans un dfSpark
df = spark.sql(query)

# transformation des dfSpark en Pandas
vict = df.toPandas()

# COMMAND ----------

# afficher un extrait des données avec head
vict.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Vehicules
# MAGIC Contenu dans le fichier veh.csv

# COMMAND ----------

# requete SQL pour sélectionner toutes les colones d'une table
query = "SELECT * FROM hive_metastore.default.vehicule"

# exécution de la requete et enregistrement dans un dfSpark
df = spark.sql(query)

# transformation des dfSpark en Pandas
veh = df.toPandas()

# COMMAND ----------

# afficher un extrait des données avec head
veh.head()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Préparation

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Importation des librairies et packages nécessaires

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import recall_score, f1_score

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Assemblage des données
# MAGIC On va merger les fichiers entre-eux en faisant attention aux identifiants tout ça en utilisant `Pandas` en se basant sur le Tuto d'Ilyes.

# COMMAND ----------

victime = vict.merge(veh,on=['Num_Acc','num_veh'])
accident = carac.merge(lieux,on = 'Num_Acc')
victime = victime.merge(accident,on='Num_Acc')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Traitement des valeurs manquantes

# COMMAND ----------

nan_values = victime.isna().sum()

nan_values = nan_values.sort_values(ascending=True)*100/127951

ax = nan_values.plot(kind='barh', figsize=(8, 10),  color='#AF7AC5', zorder=2, width=0.85)

ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

vals = ax.get_xticks()

for tick in vals:
  ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

# COMMAND ----------

nans = ['v1','v2','lartpc',
       'larrout','locp','etatp',
       'actp','voie','pr1',
       'pr','place']

victime = victime.drop(columns = nans)

# COMMAND ----------

victime = victime.dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calcul des corrélations et des variances

# COMMAND ----------

victime.corr()
victime.var()

# COMMAND ----------

# MAGIC %md
# MAGIC Le calcul de la matrice de corrélation ne donne rien. Les valeurs des corrélations étaient très loin de 1 ou -1.  
# MAGIC En revanche, le calcul de la variance a permi de montrer que la variable `an` ne variait quasiment pas. Comme elle n’apporte aucune information nous pouvons la retirer.

# COMMAND ----------

victime = victime.drop(columns=['an'])

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Encodage de hrmn, du mois et de la position GPS

# COMMAND ----------

hrmn=pd.cut(victime['hrmn'],24,labels=[str(i) for i in range(0,24)])

# COMMAND ----------

victime['hrmn']=hrmn.values

# COMMAND ----------

# On extrait du tableau la latitude et la longitude
X_lat = victime['lat']
X_long = victime['long']

# On définit tous nos points à classifier
X_cluster = np.array((list(zip(X_lat, X_long))))

# Kmeans nous donne pour chaque point la catégorie associée
clustering = KMeans(n_clusters=15, random_state=0)
clustering.fit(X_cluster)

# Enfin on ajoute les catégories dans la base d'entraînement
geo = pd.Series(clustering.labels_)
victime['geo'] = geo

# COMMAND ----------

# Suppressions des NaN values
victime = victime.dropna()

# COMMAND ----------

y = victime['grav']

features = ['catu','sexe','trajet','secu',
            'catv','an_nais','mois',
            'occutc','obs','obsm','choc','manv',
            'lum','agg','int','atm','col','gps',
            'catr','circ','vosp','prof','plan',
            'surf','infra','situ','hrmn','geo']

# COMMAND ----------

X_train_data = pd.get_dummies(victime[features].astype(str))

# COMMAND ----------

# On commence par normaliser les données :
X_train_data = normalize(X_train_data.values)

# On divise la base en bases d'entraînements et de test :
X_train, X_test, y_train, y_test = train_test_split(X_train_data,y, test_size=0.2)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Création du modèle

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### KNN pour prédire la gravité d'un accident

# COMMAND ----------

# Crée un modèle KNN
knn = KNeighborsClassifier()

# Entraîne le modèle sur les données d'entraînement
knn.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Évaluation du modèle `knn`

# COMMAND ----------

from sklearn.metrics import accuracy_score

# Prédire les étiquettes sur les données de test
y_pred = knn.predict(X_test)

# Calculer le taux de réussite du modèle
accuracy = accuracy_score(y_test, y_pred)

# Afficher le taux de réussite du modèle
print("Taux de réussite du modèle : ", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Amélioration du modèle KNN
# MAGIC En utilisant `GridSearchCV`

# COMMAND ----------

# Importer la classe GridSearchCV
from sklearn.model_selection import GridSearchCV

# Définir les hyperparamètres à tester
param_grid = {'n_neighbors': [1, 3, 5, 7, 9], 'weights': ['uniform', 'distance']}

# COMMAND ----------

# Créer un modèle KNN
knn = KNeighborsClassifier()

# Créer un objet GridSearchCV pour l'optimisation des hyperparamètres
grid_search = GridSearchCV(knn, param_grid, cv=5)

# Entraîner le modèle avec GridSearchCV sur les données d'entraînement
grid_search.fit(X_train, y_train)

# Obtenir les meilleurs hyperparamètres trouvés
best_params = grid_search.best_params_

# COMMAND ----------

# Mettre à jour le modèle avec les meilleurs hyperparamètres
knn.best_params_ = best_params

# Entraîner le modèle mis à jour sur toutes les données d'entraînement
knn.fit(X_train, y_train)

# COMMAND ----------

# Prédire les étiquettes sur les données de test avec le modèle amélioré
y_pred = knn.predict(X_test)

# Calculer le taux de réussite du modèle amélioré
accuracy = accuracy_score(y_test, y_pred)

# Afficher le taux de réussite du modèle amélioré
print("Taux de réussite du modèle amélioré : ", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### DecisionTreeClassifier pour prédire la gravité d'un accident

# COMMAND ----------

# MAGIC %md
# MAGIC Redécoupage des données pour que l'entraînnement se fasse plus rapidement

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X_train_data, y, test_size=0.9, stratify=y)

# COMMAND ----------

# Import des libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# définition des paramètres de Grid
param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_leaf': [1, 2, 3]
}

# DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()

# Create a GridSearchCV object
grid_search = GridSearchCV(tree_clf, param_grid)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# COMMAND ----------

# Import des librairies nécessaires
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Prédiction sur l'ensemble de test
y_pred = grid_search.predict(X_test)

# COMMAND ----------

# Évaluation du modèle avec taux de réussite
accuracy = accuracy_score(y_test, y_pred)

#affichage
print("Taux de réussite du modèle DecisionTreeClassifier : ", accuracy)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### RandomForest pour prédire la gravité d'un accident

# COMMAND ----------

# Import des librairies nécessaires
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Création de l'estimateur RandomForestClassifier
estimator = RandomForestClassifier()

# Création de la grille des paramètres à tester
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_leaf': [1, 3, 5]
}

# Création de l'objet GridSearchCV pour l'optimisation des paramètres
grid_search = GridSearchCV(estimator, param_grid, cv=5)

# Entraînement du modèle avec la recherche des meilleurs paramètres
grid_search.fit(X_train, y_train)

# Meilleurs paramètres trouvés
best_params = grid_search.best_params_

# COMMAND ----------

# Import des librairies nécessaires
from sklearn.metrics import accuracy_score

# Prédiction avec le modèle RandomForest
y_pred = grid_search.predict(X_test)

# Évaluation du modèle avec taux de réussite
accuracy = accuracy_score(y_test, y_pred)

# Affichage du taux de réussite du modèle RandomForestClassifier
print("Taux de réussite du modèle RandomForestClassifier : ", accuracy)

# COMMAND ----------

foretalea = grid_search

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Sauvegarde du meilleur modèle sous `BestModel`
# MAGIC Le modèle de forêt aléatoire enregistré sous `foretalea` est celui qui a le meilleur taux de réussite.  
# MAGIC Je vais l'enregistrer sous `BestModel` dans mes modèles accessibles.

# COMMAND ----------

!pip install mlflow

# COMMAND ----------

import mlflow

# COMMAND ----------

# Sauvegarder le modèle avec MLflow
mlflow.sklearn.log_model(foretalea, "BestModel")

# COMMAND ----------

# Enregister le modèle avec une version spécifique
model_uri = "runs:/3ce3e7d9b7654a33bd71df374bf09fbc/BestModel"
registered_model_name = "BestModel"
model_version = "1.0"

model_details = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name
)
