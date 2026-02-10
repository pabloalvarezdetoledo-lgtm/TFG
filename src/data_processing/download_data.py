"""Descarga de datos financieros y económicos utilizando yfinance y FRED API"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from src.utils.config import Data_Raw, Start_Date, End_Date

def download_SP500():
    """Descarga SP500 desde Yahoo"""
    sp500 = yf.download("^GSPC", start=Start_Date, end=End_Date)
    sp500.to_csv(Data_Raw / "SP500_daily.csv")
    print("SP500 descargado y guardado.")
    return sp500

def download_Blanace_Fed():
    """Descarga datos de balance de la Fed desde FRED St Louis"""
    fred = Fred(api_key = "8005f1bf90e91f01427d49ea9ed4ea41")
    balance = fred.get_series('WALCL',
                              observation_start= Start_Date,
                              observation_end= End_Date)
    balance.to_csv(Data_Raw / "Fed_Balance.csv")
    print("Balance de la Fed descargado y guardado.")
    return balance

def download_VIX():
    """Descarga VIX desde Yahoo"""
    vix = yf.download("^VIX", start=Start_Date, end=End_Date)
    vix.to_csv(Data_Raw / "VIX_daily.csv")
    print("VIX descargado y guardado.")
    return vix

def download_Treasury_Yields():
    """Descarga rendimientos de bonos del Tesoro desde FRED St Louis"""
    fred = Fred(api_key = "8005f1bf90e91f01427d49ea9ed4ea41")

    yields = pd.DataFrame({
        'DGD2': fred.get_series('DGD2', observation_start= Start_Date),
        'DGS10': fred.get_series('DGS10', observation_start= Start_Date)
    })

    yields.to_csv(Data_Raw / "Treasury_Yields.csv")
    print("Rendimientos de bonos del Tesoro descargados y guardados.")
    return yields

if __name__ == "__main__":
    download_SP500()
    download_Blanace_Fed()
    download_VIX()
    download_Treasury_Yields()
    print("\n Todos los datos han sido descargados y guardados en la carpeta 'data/raw'.") 