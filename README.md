# Databricks-Accidents-Project
TP EXAM (BUT SD3)

## Introduction du projet

Ce projet consiste en l'analyse et la prédiction de la gravité des accidents de la route à l'aide de données venant d'une étude de <b>Ilyes Talbi</b>. En utilisant des techniques de data minining, le projet vise à modéliser un classifieur capable de prédire la gravité d'un accident à partir des autres avriables accessibles dans les données.

## Sources des Données

Les données utilisées dans ce projet proviennent de fichiers CSV décrivant les accidents. Ces fichiers ont été transformés et analysés pour préparer le modèle de prédiction. Pour plus de détails sur la source originale des données, veuillez consulter les travaux de [Ilyes Talbi](https://larevueia.fr/xgboost-vs-random-forest-predire-la-gravite-dun-accident-de-la-route/).

## Éléments du Repository

Le repository contient les éléments suivants :

Notebooks `.dbc` pour l'importation des données sous `(EXAM) Importation des CSVs`.
Notebooks `.dbc` pour l'analyse exploratoire des données et la préparation des modèles de classification sous `ÀHMEDfi-EXAM`

## Travail réalisé et remarques
Parmi les différents modèles testés (`KNN`, `DecisionTreeClassifier`, `RandomForest`), le modèle `RandomForest` a été retenu pour sa supériorité en termes de taux de réussite. Ce modèle, je l'ai nommé `foretalea`, avant de l'enregistrer en tant que `BestModel`.

Endpoint du Modèle

Le modèle `BestModel` est accessible via une API, permettant ainsi une intégration facile dans des applications ou des services tiers souhaitant utiliser le modèle pour des prédictions en temps réel. Le lien de l'endpoint API est disponible dans [ici](https://adb-1959923359795803.3.azuredatabricks.net/serving-endpoints/AHMEDfi_api/invocations)
