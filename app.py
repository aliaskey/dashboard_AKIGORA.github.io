# Importation des bibliothèques
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import warnings
import os
import tempfile
import time
from streamlit_folium import st_folium
import folium
import json
import requests
from datetime import datetime
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from streamlit_lottie import st_lottie
from streamlit_extras.dataframe_explorer import dataframe_explorer

# Configuration de la page
st.set_page_config(layout="wide", page_title="AKIGORA", page_icon="👨‍")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Fonctions utilitaires
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def create_centered_markdown(text):
    st.markdown(f"<h1 style='text-align: center;'>{text}</h1>", unsafe_allow_html=True)

########################################################################################################################################################################
# Création de la page Home
def home():


    create_centered_markdown('Analyse statistique AKIGORA')

    cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1])

   
    cols[4].button("start")
    st.session_state.page = "dashboard"
            # Actions à exécuter lorsque le bouton est pressé

    # Charger le fichier JSON de l'animation
    with open('Animation - 1702075953474.json', 'r') as file:
        lottie_json = json.load(file)

    # Utiliser st.columns pour centrer l'animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:  # Placez l'animation dans la colonne du milieu
        st_lottie(lottie_json, height=500, width=700, key="animation")

#######################################################################################################################################################################
# Création de la page dashboard
def dashboard():
    # Téléchargement des datasets
    # df_company = pd.read_csv(r'df_company_VF.csv')
    # df_expert = pd.read_csv(r'df_expert_VF.csv')
    # df_intervention = pd.read_csv(r'df_intervention_VF.csv')
    # df_newsletter = pd.read_csv(r'df_newsletter_VF.csv')
    # df_recommandation = pd.read_csv(r'df_recommandation_VF.csv')
    # df_user = pd.read_csv(r'df_user_VF.csv')
    df_company = pd.read_csv(r'Collection profile (type company).csv')
    df_expert = pd.read_csv(r'Collection profile (type expert).csv')
    df_intervention = pd.read_csv(r'Collection intervention.csv')
    df_newsletter = pd.read_csv(r'Collection newsletter.csv')
    df_recommandation = pd.read_csv(r'Collection recommandation.csv')
    df_user = pd.read_csv(r'Collection user.csv')
########################################################################################################################################################################    
  
    # Création de colonnes
    col1, col2, col3 = st.columns([3,4,3])
    
    # Chargement et affichage de l'image dans la colonne centrale
    with col2:
        st.image('imgAkigora.png', use_column_width=True)
############################################################################################################################################################################    

    # Application du CSS 
    # st.markdown(custom_css, unsafe_allow_html=True)
    if st.button("Retour à la page d'accueil"):
        st.session_state.page = "home"
 
    st.markdown("<h1 style='text-align: center;'>Dashboard par département</h1>", unsafe_allow_html=True)
    st.write('Sélectionner un onglet "🔽"')
    tab1, tab2, tab3, tab4 = st.tabs(['**🔽Ressources humaines**', '**🔽Marketing**','**🔽Commerce**', '**🔽direction/Tech**'])
    with tab1: 
        # Vos titres et sous-titres
        # st.title("Dashboard d'analyse statistique AKIGORA")
        # st.header('Période de janv. 2020 au 31 déc. 2023')
        # st.subheader(f'# Base de données du département {departements}')
        # st.write('*******')
        # st.markdown('----')
        # st.subheader('Analyse graphique ')
########################################################################################
# # Filtre par an        
   
        # Charger le DataFrame
        df_expert = pd.read_csv(r'Collection profile (type expert).csv')
        
        # Créer une liste d'années
        annees = list(range(2018, datetime.now().year + 1))
        
        # Menu déroulant pour la sélection de l'année
        selected_year = st.sidebar.selectbox("Sélectionner l'année pour département RH", annees)
       
        # Fonction pour filtrer le DataFrame en fonction de l'année
        def filter_df_by_year(df, selected_year, date_column):
            df_copy = df.copy()
            df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')  # Convertit les dates invalides en NaT
            df_copy = df_copy.dropna(subset=[date_column])  # Supprime les lignes avec des dates invalides
            df_copy['annee'] = df_copy[date_column].dt.year  # Extraction de l'année
            return df_copy[df_copy['annee'] == selected_year]
        
        # Appliquer le filtre au DataFrame concerné
        df_expert_filtered = filter_df_by_year(df_expert, selected_year, 'createdAt')


###########################################################################################        
#Onglet RH
        # Nombre d'experts inscrits sur la plateforme :
        nb_experts_inscrits = df_expert[df_expert['type'] == 'expert'].shape[0]
        #Nombre d'experts désinscrits :
        nb_experts_desinscrits = df_user[df_user['email'] == 'removed@akigora.com'].shape[0]
        #Nombre d'experts visibles sur la plateforme :
        nb_experts_visibles = df_expert[(df_expert['type'] == 'expert') & 
                                (df_expert['visible'] == True) & 
                                (df_expert['done'] == True) & 
                                (df_expert['isFake'] == False) & 
                                (df_expert['temporarilyInvisible'] == False)].shape[0]
        #Nombre de recommandations et pourcentage d'experts recommandés :
        total_recommendations = df_recommandation.shape[0]
        total_experts_recommended = df_recommandation['expertId'].nunique()
        percent_experts_recommended = (total_experts_recommended / nb_experts_inscrits) * 100

        # Affichage des KPI dans Streamlit
        st.markdown("<h3 style='text-align: center;'>KPI du département des ressources humaines</h3>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Experts Inscrits", nb_experts_inscrits)
        with col2:
            st.metric("Experts Désinscrits", nb_experts_desinscrits)
        with col3:
            st.metric("Experts Visibles", nb_experts_visibles)
        with col4:
            st.metric("Nombre Total de Recommandations", total_recommendations) 
###################################################################################################        
        #Pourcentage d'experts à profil complété à 100% :
        pourcentage_profil_complet = df_expert[(df_expert['type'] == 'expert') & (df_expert['percentage'] == 100)].shape[0] /df_expert[df_expert['type'] == 'expert'].shape[0] * 100
        #Pourcentage d'experts à profil non complété :
        pourcentage_profil_non_complet = 100 - pourcentage_profil_complet
        #Pourcentage d'experts sans référence :
        pourcentage_sans_reference = df_expert[(df_expert['type'] == 'expert') & (df_expert['references'].isna() | df_expert['references'] == '')].shape[0] / df_expert[df_expert['type'] =='expert'].shape[0] * 100
        #Pourcentage d'experts qui ont utilisé LinkedIn pour leur profil :
        pourcentage_linkedin = df_expert[df_expert['linkedInImport'].notna()].shape[0] / df_expert[df_expert['type'] == 'expert'].shape[0] * 100
############################################################################################################################"
# Affichage 1
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pourcentage de Profils Complétés", f"{pourcentage_profil_complet:.2f}%")
        with col2:
            st.metric("Pourcentage de Profils Non Complétés", f"{pourcentage_profil_non_complet:.2f}%")
        with col3:
            st.metric("Pourcentage d'Experts Utilisant LinkedIn", f"{pourcentage_linkedin:.2f}%")
        with col4:
            st.metric("Pourcentage d'Experts Recommandés", f"{percent_experts_recommended:.2f}%")    
####################################################################################################################################################################
# calcul des indicateurs    
        # Affichage des métriques et des données
        def display_metrics_and_data(df_expert, df_recommandation):
            total_experts = df_expert[df_expert['type'] == 'expert'].shape[0]
            total_experts_recommended = df_recommandation['expertId'].nunique()
            percent_experts_recommended = (total_experts_recommended / total_experts) * 100 if total_experts else 0
            total_recommendations = df_recommandation.shape[0]
            percent_linkedin_used = 0
            if 'linkedInImport' in df_expert.columns:
                linkedin_used = df_expert[df_expert['linkedInImport'].notna()].shape[0]
                percent_linkedin_used = (linkedin_used / total_experts) * 100 if total_experts else 0
############################################################################################################################################
#Afficher/Masquer les experts à profil non complété
            # Initialiser une variable de l'état de session pour suivre si les données doivent être affichées
            if 'show_experts' not in st.session_state:
                st.session_state.show_experts = False
            
            # Lorsque le bouton est cliqué, basculer l'état de 'show_experts'
            if st.button("Afficher/Masquer les experts à profil non complété"):
                st.session_state.show_experts = not st.session_state.show_experts
            
            # Afficher les données si 'show_experts' est True
            if st.session_state.show_experts:
                df_experts_incomplets = df_expert[(df_expert['type'] == 'expert') & (df_expert['percentage'] < 100)]
                st.write(df_experts_incomplets)
#######################################################################################################################################################
# Calcul des 10 domaines d'expertise les plus fréquents
        def plot_expertise_distribution(df_expert):
            # Calcul des 10 domaines d'expertise les plus fréquents
            expertise_counts = df_expert['domains'].value_counts().head(10)
            # Configuration de la figure pour le diagramme circulaire
            plt.figure(figsize=(6, 12))
            plt.pie(expertise_counts, labels=expertise_counts.index, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')
            st.pyplot(plt)
#####################################################################################################################################################
#carte du monde
        # Fonction pour extraire les coordonnées de la colonne 'geo'
        def extract_coordinates_from_geo(geo_data):
            if isinstance(geo_data, str):
                try:
                    geo_json = json.loads(geo_data)
                    if 'location' in geo_json and 'coordinates' in geo_json['location']:
                        # Les coordonnées sont dans un tableau avec [longitude, latitude]
                        return geo_json['location']['coordinates'][1], geo_json['location']['coordinates'][0]
                except json.JSONDecodeError:
                    pass
            return None, None
        
        # Appliquer la fonction pour créer les colonnes 'latitude' et 'longitude' dans df_expert
        df_expert = pd.read_csv(r'Collection profile (type expert).csv')  # Remplacez par le chemin correct vers votre fichier
        df_expert['latitude'], df_expert['longitude'] = zip(*df_expert['geo'].apply(extract_coordinates_from_geo))
        
        # Créer un nouveau DataFrame contenant uniquement les colonnes 'latitude' et 'longitude'
        df_for_map = df_expert[['latitude', 'longitude']].dropna()
# 
        #Affichage 2 de la page RH       
        # Appel des fonctions dans l'application Streamlit
        display_metrics_and_data(df_expert, df_recommandation)
#############################################################################################################################     # Affichage 2 
        # Création de la mise en page avec la carte et le graphique Matplotlib côte à côte
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='text-align: center;'>Carte des Experts</h3>", unsafe_allow_html=True)
            st.map(df_for_map)  # Utilisation de st.map pour afficher la carte
        
        with col2:
            # Appel de la fonction pour afficher le graphique Matplotlib
            st.markdown("<h3 style='text-align: center;'>Top 10 des Domaines d’Expertise</h3>", unsafe_allow_html=True)
            plot_expertise_distribution(df_expert)

########################################################################################################################################################           
#Onglet MARKETING 
    with tab2:
        st.markdown("<h3 style='text-align: center;'>KPI du département Marketing</h3>", unsafe_allow_html=True)
######################################################
#liste
        # Initialiser une variable de l'état de session
        if 'show_profiles' not in st.session_state:
            st.session_state.show_profiles = False
        
        # Bouton pour afficher/masquer les profils les plus consultés
        if st.button('Afficher/Masquer les profils les plus consultés'):
            st.session_state.show_profiles = not st.session_state.show_profiles
        
        # Afficher les données si 'show_profiles' est True
        if st.session_state.show_profiles:
            profils_populaires = df_expert.nlargest(10, 'profile_views')  # Afficher les 10 profils les plus consultés
            st.write("Profils les plus consultés :")
            st.table(profils_populaires[['profile_name', 'profile_views']])  # Remplacez 'profile_name' par le nom de la colonne appropriée
 
#################################################
        pourcentage_abonnes_newsletter = (df_newsletter.shape[0] / df_user.shape[0]) * 100
        st.metric("Pourcentage d'Abonnés Newsletter", f"{pourcentage_abonnes_newsletter:.2f}%")
        
        repartition_diplome = df_expert['studyTitle'].value_counts(normalize=True) * 100
        # st.bar_chart(repartition_diplome)
        
        repartition_type_structure = df_company['company.type'].value_counts(normalize=True) * 100
        # st.bar_chart(repartition_type_structure)
        repartition_experience = df_expert['experienceTime'].value_counts(normalize=True) * 100
        # st.bar_chart(repartition_experience)
###################################################################

        # repartition_diplome = df_expert['studyTitle'].value_counts(normalize=True) * 100
        # st.bar_chart(repartition_diplome)
        
        
        # Titres d'Intervention les Plus Fréquents
        intitules_populaires = df_intervention['intitule'].value_counts().head(10)
        st.write("Titres d'Intervention les Plus Fréquents")
        

        # Affichage des KPI dans Streamlit
        col1, col2, col3 = st.columns(3)
        with col1:
            st.bar_chart(repartition_diplome)
        with col2:
            st.bar_chart(repartition_experience)
        with col3:
            st.write("Titres d'Intervention les Plus Fréquents")
            st.table(intitules_populaires)

###################################################################################################""""                   
#onglet commerce    
    with tab3:

        nb_missions = df_intervention.shape[0]
        taux_journalier_min = df_expert['daily_hourly_prices.daily_price_min'].min()
        taux_journalier_max = df_expert['daily_hourly_prices.daily_price_max'].max()
        taux_horaire_moyen = df_expert[['daily_hourly_prices.hourly_price_min', 'daily_hourly_prices.hourly_price_max']].mean().mean().round()
        nb_clients_ecoles = df_user[df_user['companyOrSchool'] == 'school'].shape[0]
        nb_clients_entreprises = df_user[df_user['companyOrSchool'] == 'company'].shape[0]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Taux Journalier Minimum", taux_journalier_min)
        with col2:
            st.metric("Taux Journalier Maximum", taux_journalier_max)    
        with col3:    
            st.metric("Nombre de Clients Écoles", nb_clients_ecoles)
                
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Taux Horaire Moyen", taux_horaire_moyen)
        with col2:
            st.metric("Nombre de Clients Entreprises", nb_clients_entreprises)  
        with col3:    
            st.metric("Nombre de Missions", nb_missions)

########################################################################################################"""
#Onglet Direction/tech
        
    
    with tab4:
        # Vos titres et sous-titres
        # st.title("Dashboard d'analyse statistique AKIGORA")
        # st.header('Période de janv. 2020 au 31 déc. 2023')
        st.subheader(f'Base de données par département')
        # st.write('*******')
        # st.markdown('----')
        st.subheader('Exploration de la base de données')
            # df_explore = dataframe_explorer(user, case=False)
            # st.dataframe(df_explore, use_container_width=True)
            # # Afficher les statistiques descriptives selon région sélectionnée
        st.write('#### Statistiques descriptives')
        st.write(df_intervention, df_user, df_expert, df_company, df_recommandation, df_newsletter)
    
########################################################################################################################
        
        # Création d'un histogramme pour les heures planifiées par mission
        plt.figure(figsize=(10, 6))
        df_intervention['hours_planned'] = pd.to_numeric(df_intervention['hours_planned'], errors='coerce')
        df_intervention.dropna(subset=['hours_planned'], inplace=True)
        plt.hist(df_intervention['hours_planned'], bins=30, color='skyblue', edgecolor='black')
        plt.title('Distribution des heures planifiées par mission')
        plt.xlabel('Heures')
        plt.ylabel('Nombre de missions')
        plt.grid(True)
        plt.show()
        
################################################################################################################"


# Initialiser l'état de la page si ce n'est pas déjà fait
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Afficher la page appropriée en fonction de l'état
if st.session_state.page == "home":
    home()
elif st.session_state.page == "dashboard":
    dashboard()
   
    # dans le CMD saisir : streamlit run correction6_my_streamlit_app.py   