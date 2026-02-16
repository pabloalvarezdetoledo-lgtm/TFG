"""
Script de descarga de datos financieros
Descarga series desde FRED, Yahoo Finance y fuentes externas.
Guarda archivos CSV en data/raw/

Autor: Pablo Álvarez de Toledo Rodríguez
Fecha: Enero 2026
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import requests
from tqdm import tqdm

# Añadir src/ al path para poder importar config
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import (
    DATA_RAW,
    DATA_EXTERNAL,
    START_DATE,
    END_DATE,
    FRED_SERIES,
    YAHOO_TICKERS,
    EXTERNAL_SOURCES
)


# =============================================================================
# CONFIGURACIÓN INICIAL
# =============================================================================

# Cargar variables de entorno desde .env
load_dotenv()

# Obtener API key de FRED
FRED_API_KEY = os.getenv('FRED_API_KEY')

if not FRED_API_KEY:
    raise ValueError(
        "No se encontró FRED_API_KEY en archivo .env\n"
        "Por favor, crea un archivo .env en la raíz del proyecto con:\n"
        "FRED_API_KEY=tu_clave_aqui"
    )

# Inicializar cliente de FRED
fred = Fred(api_key=FRED_API_KEY)


# =============================================================================
# FUNCIONES DE DESCARGA
# =============================================================================

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
    
    for name, fred_code in tqdm(series_dict.items(), desc="Series FRED"):
        try:
            # Descargar serie
            series = fred.get_series(
                fred_code,
                observation_start=START_DATE,
                observation_end=END_DATE
            )
            
            # Convertir a DataFrame
            df = pd.DataFrame({
                'date': series.index,
                name: series.values
            })
            
            # Información sobre la serie
            info = fred.get_series_info(fred_code)
            frequency = info.get('frequency_short', 'Unknown')
            units = info.get('units', 'Unknown')
            
            # Guardar como CSV
            output_path = DATA_RAW / f"fred_{name}.csv"
            df.to_csv(output_path, index=False)
            
            # Almacenar en diccionario
            downloaded_data[name] = df
            
            print(f"  ✓ {name:20s} | Código: {fred_code:15s} | "
                  f"Freq: {frequency:10s} | Obs: {len(df):5d}")
            
        except Exception as e:
            print(f"  ✗ {name:20s} | ERROR: {str(e)}")
            continue
    
    return downloaded_data


def download_yahoo_series(tickers_dict):
    """
    Descarga series desde Yahoo Finance y las guarda en data/raw/
    
    Parameters
    ----------
    tickers_dict : dict
        Diccionario con formato {'nombre': 'TICKER'}
        Ejemplo: {'sp500': '^GSPC'}
    
    Returns
    -------
    dict
        Diccionario con los DataFrames descargados
        
    Notes
    -----
    - Descarga precios diarios (Open, High, Low, Close, Volume, Adj Close)
    - Guarda solo la columna 'Adj Close' o 'Close' renombrada como el nombre de la serie
    - Se guarda como CSV con nombre: yahoo_{nombre}.csv
    """
    print("\n" + "="*70)
    print("DESCARGANDO SERIES DE YAHOO FINANCE")
    print("="*70)
    
    downloaded_data = {}
    
    for name, ticker in tqdm(tickers_dict.items(), desc="Series Yahoo"):
        try:
            # Descargar datos usando Ticker object (más robusto)
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(
                start=START_DATE,
                end=END_DATE,
                auto_adjust=True  # Automáticamente ajusta por splits/dividendos
            )
            
            # Verificar que data no esté vacío
            if data.empty:
                raise ValueError(f"No se obtuvieron datos para {ticker}")
            
            # Con auto_adjust=True, la columna se llama 'Close' (ya ajustada)
            if 'Close' in data.columns:
                price_series = data['Close']
            else:
                raise ValueError(f"Columna 'Close' no encontrada en {ticker}")
            
            # Crear DataFrame limpio
            df = pd.DataFrame({
                'date': price_series.index,
                name: price_series.values
            })
            
            # Resetear índice
            df = df.reset_index(drop=True)
            
            # Eliminar NaN
            df = df.dropna()
            
            # Verificar que tenemos datos
            if len(df) == 0:
                raise ValueError(f"DataFrame vacío después de limpiar NaN")
            
            # Guardar como CSV
            output_path = DATA_RAW / f"yahoo_{name}.csv"
            df.to_csv(output_path, index=False)
            
            # Almacenar en diccionario
            downloaded_data[name] = df
            
            print(f"  ✓ {name:20s} | Ticker: {ticker:10s} | Obs: {len(df):5d}")
            
        except Exception as e:
            print(f"  ✗ {name:20s} | ERROR: {str(e)}")
            # Imprimir más detalles para debugging
            import traceback
            print(f"     Detalles: {traceback.format_exc()}")
            continue
    
    return downloaded_data

def download_shiller_cape():
    """
    Descarga dataset de Shiller (CAPE, Earnings, Dividends)
    
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: date, price, dividend, earnings, cape
        
    Notes
    -----
    El archivo de Shiller es un Excel (.xls) con formato peculiar:
    - Las primeras filas son encabezados descriptivos
    - Los datos empiezan en la fila 8 (aproximadamente)
    - Contiene datos mensuales desde 1871
    
    Columnas principales:
    - Date: formato YYYY.MM (ejemplo: 2024.01 para enero 2024)
    - P: Precio del S&P 500
    - D: Dividendo
    - E: Earnings (ganancias)
    - CAPE: Cyclically Adjusted Price-Earnings Ratio (Shiller P/E)
    
    Referencias
    ----------
    - Fuente: http://www.econ.yale.edu/~shiller/data.htm
    - Paper: Shiller, R.J. (2015). Irrational Exuberance, 3rd Edition.
    """
    print("\n" + "="*70)
    print("DESCARGANDO DATASET DE SHILLER (CAPE)")
    print("="*70)
    
    try:
        url = EXTERNAL_SOURCES['shiller_cape']
        
        # Descargar archivo Excel
        print(f"  Descargando desde: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Guardar temporalmente
        temp_file = DATA_EXTERNAL / "shiller_cape_raw.xls"
        with open(temp_file, 'wb') as f:
            f.write(response.content)
        
        # Leer Excel (skiprows para saltar encabezados)
        # Nota: El formato puede cambiar ligeramente con actualizaciones de Shiller
        df_raw = pd.read_excel(
            temp_file,
            sheet_name='Data',
            skiprows=7  # Saltar encabezados descriptivos
        )
        
        # Renombrar columnas (los nombres varían, usar posiciones es más robusto)
        # Columnas típicas: Date, P, D, E, CPI, CAPE, etc.
        df = df_raw.iloc[:, [0, 1, 2, 3, 10]].copy()  # Date, P, D, E, CAPE
        df.columns = ['date_raw', 'price', 'dividend', 'earnings', 'cape']
        
        # Convertir fecha de formato YYYY.MM a datetime
        # Ejemplo: 2024.01 → 2024-01-01
        df['year'] = df['date_raw'].astype(str).str.split('.').str[0].astype(int)
        df['month'] = df['date_raw'].astype(str).str.split('.').str[1].astype(int)
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        
        # Seleccionar columnas finales
        df = df[['date', 'price', 'dividend', 'earnings', 'cape']]
        
        # Filtrar por rango de fechas
        df = df[
            (df['date'] >= pd.to_datetime(START_DATE)) &
            (df['date'] <= pd.to_datetime(END_DATE))
        ]
        
        # Limpiar valores inválidos (Shiller a veces tiene "NA" como string)
        df = df.replace('NA', pd.NA)
        df = df.dropna()
        
        # Guardar CSV procesado
        output_path = DATA_EXTERNAL / "shiller_cape.csv"
        df.to_csv(output_path, index=False)
        
        print(f"  ✓ Shiller CAPE     | Obs: {len(df):5d} | "
              f"Periodo: {df['date'].min().strftime('%Y-%m')} a "
              f"{df['date'].max().strftime('%Y-%m')}")
        
        # Borrar archivo temporal
        temp_file.unlink()
        
        return df
        
    except Exception as e:
        print(f"  ✗ ERROR descargando Shiller CAPE: {str(e)}")
        print("  Nota: El formato del archivo de Shiller puede haber cambiado.")
        print("  Descarga manual desde: http://www.econ.yale.edu/~shiller/data.htm")
        return None


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Ejecuta pipeline completo de descarga de datos
    """
    print("\n" + "="*70)
    print(" PIPELINE DE DESCARGA DE DATOS FINANCIEROS")
    print("="*70)
    print(f"Periodo: {START_DATE} a {END_DATE}")
    print(f"Destino: {DATA_RAW}")
    print("="*70)
    
    # Timestamp de inicio
    start_time = datetime.now()
    
    # 1. Descargar series de FRED
    fred_data = download_fred_series(FRED_SERIES)
    
    # 2. Descargar series de Yahoo Finance
    yahoo_data = download_yahoo_series(YAHOO_TICKERS)
    
    # 3. Descargar dataset de Shiller
    shiller_data = download_shiller_cape()
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN DE DESCARGA")
    print("="*70)
    print(f"Series FRED descargadas:  {len(fred_data)}/{len(FRED_SERIES)}")
    print(f"Series Yahoo descargadas: {len(yahoo_data)}/{len(YAHOO_TICKERS)}")
    print(f"Shiller CAPE:             {'✓' if shiller_data is not None else '✗'}")
    print(f"\nTiempo total: {(datetime.now() - start_time).total_seconds():.1f}s")
    print(f"Archivos guardados en: {DATA_RAW}")
    print("="*70)
    
    # Listar archivos creados
    print("\nArchivos creados:")
    for file in sorted(DATA_RAW.glob("*.csv")):
        size_kb = file.stat().st_size / 1024
        print(f"  - {file.name:40s} ({size_kb:>8.1f} KB)")
    
    if (DATA_EXTERNAL / "shiller_cape.csv").exists():
        file = DATA_EXTERNAL / "shiller_cape.csv"
        size_kb = file.stat().st_size / 1024
        print(f"  - {file.name:40s} ({size_kb:>8.1f} KB)")
    
    print("\n✓ Descarga completada exitosamente\n")


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    main()