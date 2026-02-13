"""
Script de descarga de datos financieros
Descarga series desde FRED, Yahoo Finance y fuentes externas.
Guarda archivos CSV en data/raw/
"""
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import requests
from tqdm import tqdm
# Cargar configuración y variables de entorno
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import (
    DATA_RAW,
    DATA_EXTERNAL,
    START_DATE,
    END_DATE,
    FRED_SERIES,
    YAHOO_TICKERS,
    EXTERNAL_SCOURCES 
)

#CONFIGURACIÓN INICIAL 

#Carga de variables de entorno
load_dotenv()

#obtención API key de FRED
FRED_API_KEY = os.getenv("FRED_API_KEY")

if not FRED_API_KEY:
    raise ValueError(
        "No se encontró la clave de API de FRED. Por favor, asegúrate de tener un archivo .env con la variable FRED_API_KEY definida"
    )

fred = Fred(api_key = FRED_API_KEY)

#Funciones de Descargas

def download_fred_series(series_dict):
    """
    Descarga series desde FRED y las guarda en data/raw/
    
    Parameters
    ----------
    series_dict : dict
        Diccionario con formato {'nombre': 'CODIGO_FRED'}
        Ejemplo: {'fed_balance': 'WALCL'}
    
    Returns
    -------
    dict
        Diccionario con los DataFrames descargados
        
    Notes
    -----
    - Las series se descargan en su frecuencia original (diaria, semanal, mensual)
    - Se guardan como CSV con nombre: fred_{nombre}.csv
    - Si una serie no está disponible, se salta y se imprime warning
    """
    print("\n" + "="*70)
    print("DESCARGANDO SERIES DE FRED")
    print("="*70)
    
    downloaded_data = {}

    for name, fred_code in tqdm(series_dict.items(), desc = "Series FRED"):
        try:
            series = fred.get_series(
                fred_code,
                observation_start = START_DATE,
                observation_end = END_DATE
            )
            df = pd.DataFrame({
                'date': series.index,
                name: series.values
            })

            info = fred.get_series_info(fred_code)
            frequency = info.get('frequancy_short', 'Unknown')
            units = info.get('units', 'Unknown')

            output_path = DATA_RAW / f"fred_{name}.csv"
            df.to_csv(output_path, index=False)

            downloaded_data[name] = df

            print(f"  ✓ {name:20s} | Código: {fred_code:15s} | )