AKIGORA - Application Streamlit
Introduction
Cette application Streamlit, nommée AKIGORA, est conçue pour fournir une analyse statistique et des visualisations interactives. Elle intègre diverses bibliothèques Python pour le traitement des données et la création de graphiques, rendant les données facilement accessibles et compréhensibles.

Installation
Pour exécuter cette application, vous devez installer les bibliothèques nécessaires. Utilisez la commande suivante pour installer toutes les dépendances :
Saisie CMD :
pip install numpy pandas matplotlib seaborn plotly streamlit folium sklearn
Importation des Bibliothèques
Les bibliothèques suivantes sont utilisées dans cette application :

numpy et pandas pour la manipulation des données.
matplotlib, seaborn, et plotly pour la visualisation des données.
streamlit pour créer l'application web.
folium pour les cartes interactives.
sklearn pour les algorithmes d'apprentissage automatique.
D'autres utilitaires comme datetime, warnings, os, tempfile, time, json, requests.
Configuration de la Page Streamlit
La configuration de la page est définie pour un affichage large avec un titre "AKIGORA" et une icône personnalisée.

Fonctions Utilitaires
Des fonctions utilitaires sont définies pour charger des fichiers JSON, créer des textes centrés en Markdown, etc.

Pages Principales
L'application comporte deux pages principales :

Home : Cette page présente l'analyse statistique AKIGORA avec une animation Lottie et un bouton pour naviguer vers le tableau de bord.
Dashboard : Cette page offre des visualisations détaillées et des KPIs pour différents départements comme les ressources humaines, le marketing, le commerce, et la direction/technologie.
Explication du Dashboard
Le tableau de bord est organisé en plusieurs onglets, chacun se concentrant sur un département spécifique. Chaque onglet affiche des métriques clés, des graphiques et des tableaux pertinents pour une analyse approfondie.

Onglet Ressources Humaines
Filtre par année pour analyser les données spécifiques.
Métriques sur le nombre d'experts inscrits, désinscrits, visibles, et le nombre de recommandations.
Visualisations de la répartition des profils et des domaines d'expertise.

Onglet Marketing
Analyse des profils les plus consultés et des données de newsletter.
Visualisations des titres d'intervention fréquents et des statistiques sur l'expérience des experts.

Onglet Commerce
Métriques sur les missions, les taux journaliers et horaires, et le nombre de clients (écoles et entreprises).

Onglet Direction/Tech
Exploration de bases de données par département avec des statistiques descriptives.
Histogramme des heures planifiées par mission.
Exécution de l'Application
Pour exécuter l'application, utilisez la commande suivante dans le terminal :
Saisie CMD :
streamlit run votre_fichier_app.py
