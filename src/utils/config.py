"""Configuración Global del Proyecto"""
from pathlib import Path

#Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW= PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"
RESULTS_MODELS = PROJECT_ROOT / "results" / "models"

for path in [DATA_PROCESSED, DATA_EXTERNAL, DATA_RAW, RESULTS_TABLES, RESULTS_FIGURES, RESULTS_MODELS]:
    path.mkdir(parents = True, exist_ok = True)

#Códigos de series de FRED
FRED_SERIES = {
    'fed_balance': 'WALCL',                   # Total Assets, All Federal Reserve Banks (Balance de la FED)
    'ff_rate': 'DFF',                         #Federal Funds Effective Rate
    'treasury_2y': 'DGS2',                    # 2 year Treasury Constant Maturiry Rate
    'treasury_10y': 'DGS10',                  # 10 year Treasury Constant Maturiry Rate
    'spread_bbb': 'BAMLCOA4CBBB',             # ICE BofA BBB US Corporate Index OAS
    'gdp_nominal': 'GDP'                      # GDP (Trimestral)
}

# tickers de yahoo
YAHOO_TICKERS = {
    'sp500': '^GSPC',                         #S&P 500 Index
    'vix': '^VIX'                             #CBOE Volatility index
}

# Fuentes externas (URLs de descarga)
EXTERNAL_SCOURCES = {
    #Earnings de Schiler (cCAPE dataset)
    'schillr_cape': 'http://www.econ.yale.edu/~shiller/data/ie_data.xls'
}

# Parámetros de Análisis
START_DATE = "2000-01-01"
END_DATE = "2025-12-31"
MONTHLY_FREQ = "M"

#Congiuración VECM
VECM_LAG_ORDER = 2
VECM_DET_ORER = 0

#Configuración HMM
HMM_N_STATES = 2
HMM_N_ITTER = 1000
HMM_RANDOM_SEED = 42
HHM_COVARIANCE_TYPE = 'full'

#Configuración XGBoost
XGBOOST_PARAMS = {
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample': 0.8,
    'random_state': 42,
    'objective': 'reg:squarederror'
}

    #Configuración de validación cruzada para XGBoost
XGEBOOST_CV_FOLD = 5
XGBOOST_TEST_SIZE = 24

#Eventos a estudiar (Event Study)
EVENTS = {
    'QE1_announcement': '2008-11-25',           # Primera ronda QE (Bernanke)
    'QE2_announcement': '2010-11-03',           # Segunda ronda QE
    'Operation_Twist': '2011-09-21',            # Operación Twist (compra largo, vende corto)
    'QE3_announcement': '2012-09-13',           # Tercera ronda QE (open-ended)
    'Taper_tantrum': '2013-05-22',              # Bernanke anuncia reducción gradual de compras
    'Taper_begins': '2013-12-18',               # Inicio efectivo del tapering
    'COVID_crisis': '2020-03-11',               # OMS declara pandemia
    'COVID_QE_unlimited': '2020-03-15',         # Fed anuncia QE ilimitado (domingo)
    'First_rate_hike': '2022-03-16',            # Primera subida de tipos post-COVID
    'SVB_collapse': '2023-03-10',               # Colapso Silicon Valley Bank
}

# Parametros de visualización
PLOT_STYLE = 'seaborn-darkgrid'
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

    #Colores regímenes HMM
REGIMES_COLORS = {
    'bear': '#d62728',
    'bull': '#2ca02c'
}

#Validación de Configuración
def validate_config():
    """
    Valida que la configuración sea consistente.
    Se ejecuta al importar el módulo.
    """
    # Verificar que fechas sean válidas
    from datetime import datetime
    try:
        datetime.strptime(START_DATE, '%Y-%m-%d')
        datetime.strptime(END_DATE, '%Y-%m-%d')
    except ValueError as e:
        raise ValueError(f"Formato de fecha inválido: {e}")
    
    # Verificar que END_DATE > START_DATE
    if START_DATE >= END_DATE:
        raise ValueError("END_DATE debe ser posterior a START_DATE")
    
    # Verificar parámetros de modelos
    if HMM_N_STATES < 2:
        raise ValueError("HMM debe tener al menos 2 estados")
    
    if VECM_LAG_ORDER < 1:
        raise ValueError("VECM debe tener al menos 1 lag en diferencias")
    
    print("✓ Configuración validada correctamente")


# Ejecutar validación al importar
validate_config()