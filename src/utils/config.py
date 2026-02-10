"""Configuración Global del Proyecto"""
from pathlib import Path

#Rutas del proyecto
Project_Root = Path(__file__).parent.parent.parent
Data_Raw = Project_Root / "data" / "raw"
Data_Processed = Project_Root / "data" / "processed"
Results_Tables = Project_Root / "results" / "tables"
Results_Figures = Project_Root / "results" / "figures"
Results_Models = Project_Root / "results" / "models"

# Parámetros de Análisis
Start_Date = "2000-01-01"
End_Date = "2025-12-31"
Monthly_Freq = "M"

#Congiuración VECM
VECM_lag_order = 2

#Configuración HMM
HMM_N_States = 2
HMM_N_Itter = 1000
HMM_Random_Seed = 42

#Configuración XGBoost
XGBoost_Params = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'random_state': 42
}

#Eventos a estudiar (Event Study)
Events = {
    'QE1': '2008-11-25',
    'Taper_tantrum': '2013-05-22',
    'COVID19': '2020-03-11',
    'Fed_Rate_Hike': '2022-03-16'
}
