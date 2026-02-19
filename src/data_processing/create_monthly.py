"""
Script de agregación de datos a frecuencia mensual
Lee archivos CSV de data/raw/ y crea dataset mensual unificado
"""
import sys 
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent  # src/data_processing/
SRC_DIR = SCRIPT_DIR.parent                    # src/
PROJECT_ROOT = SRC_DIR.parent                  # raíz del proyecto

# Añadir src/ al path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.config import (
    DATA_RAW,
    DATA_EXTERNAL,
    DATA_PROCESSED,
    START_DATE,
    END_DATE,
    MONTHLY_FREQ
)

#FUNCIONES AUXILIARES
def resample_to_monthly(df, date_col = 'date', method = 'last'):
    """
    Agrega una serie temporal a frecuencia mensual
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columna de fecha y una o más columnas de datos
    date_col : str
        Nombre de la columna de fecha
    method : str
        Método de agregación: 'last' (último valor), 'mean' (promedio), 'sum' (suma)
    
    Returns
    -------
    pd.DataFrame
        DataFrame con frecuencia mensual (último día del mes)
        
    Notes
    -----
    Para variables de stock (precios, niveles): usar 'last'
    Para variables de flujo (retornos, crecimientos): usar 'mean' o 'sum'
    """
    df[date_col] = pd.to_datetime(df[date_col], utc = True).dt.tz_localize(None)

    df_indexed = df.set_index(date_col)

    df_filled = df_indexed.ffill()

    #Resample a mensual
    if method == 'last':
        df_monthly = df_filled.resample('ME').last()
    elif method == 'mean':
        df_monthly = df_filled.resample('ME').mean()
    elif method == 'sum':
        df_monthly.resample('ME').sum
    else:
        raise ValueError(f"Método '{method}' no reconocido. Usar: 'last', 'mean', 'sum'")
    
    # Reset índice para tener Date como Columna
    df_monthly = df_monthly.reset_index()
    df_monthly = df_monthly.rename(columns = {'date': date_col})
    
    return df_monthly

def load_and_resample(filepath, date_col = 'date', value_col=None, method = 'last'):
  """
    Carga CSV y agrega a frecuencia mensual
    
    Parameters
    ----------
    filepath : Path
        Ruta al archivo CSV
    date_col : str
        Nombre de columna de fecha
    value_col : str or None
        Nombre de columna de datos. Si None, usa todas excepto date_col
    method : str
        Método de agregación
    
    Returns
    -------
    pd.DataFrame
        DataFrame mensual
    """
#Lectura del CSV 
  df = pd.read_csv(filepath, parse_dates=[date_col])

# Selección de columnas relevantes
  if value_col is not None:
      df = df[[date_col, value_col]]
  
  # Agregar a Mensual
  df_monthly = resample_to_monthly(df, date_col=date_col, method=method)

  return df_monthly

#Función Principal de Carga
def load_all_data():
    """
    Carga todas las series desde data/raw/  y data/exteral/
    
    Returns
    --------
    dict
        Diccionario con DataFrames mensuales por categoría
    """
    print("\n" + "="*70)
    print("Cargando y Agregando Datos a Frecuancia Mensual")
    print("="*70)

    data = {}

    # 1. S&P 500 (diario -> Mensual, último valor)
    print("\n[1/9] S&P 500...")
    sp500_path = DATA_RAW / "yahoo_sp500.csv"
    if sp500_path.exists():
        df_sp500 = load_and_resample(sp500_path, value_col='sp500', method='last')
        data['sp500'] = df_sp500
        print(f" ✓ S&P 500: {len(df_sp500)} meses")
    else:
        print(f"  ✗ Archivo no encontrado: {sp500_path}")
        data['sp500'] = None

    # 2. VIX (diario -> Mensual, último valor)
    print("\n[2/9] VIX...")
    vix_path = DATA_RAW / "yahoo_vix.csv"
    if vix_path.exists():
        df_vix = load_and_resample(vix_path, value_col = 'vix', method='last')
        data['vix'] = df_vix
        print(f" ✓ VIX: {len(df_vix)} meses")
    else:
        print(f"  ✗ Archivo no encontrado: {vix_path}")
        data['vix'] = None

    # 3. Balance de la Fed (semanal -> mensual, último valor)
    print("\n[3/9] Balance de la FED...")
    balance_path = DATA_RAW / "fred_fed_balance.csv"
    if balance_path.exists():
        df_balance = load_and_resample(balance_path, value_col= 'fed_balance', method= 'last')
        data['fed_balance'] = df_balance
        print(f" ✓ Balance FED: {len(df_balance)} meses")
    else:
        print(f"  ✗ Archivo no encontrado: {balance_path}")
        data['fed_balance'] = None

    # 4. Federal Fund Rate (diario -> mensual, último valor)
    print("\n[4/9] Federal Fund Rate...")
    ff_path = DATA_RAW / "fred_ff_rate.csv"
    if ff_path.exists():
        df_ff = load_and_resample(ff_path, value_col = 'ff_rate', method = 'last')
        data['ff_rate'] = df_ff
        print(f" ✓  FF Rate: {len(df_ff)} meses")
    else:
        (f"  ✗ Archivo no encontrado: {ff_path}")
        data['ff_rate'] = None

    # 5. Treasury 2y (diario -> mensual, último valor)
    print("\n[5/9] Treasury Bonds 2 year rate...")
    t2y_path = DATA_RAW / "fred_treasury_2y.csv"
    if t2y_path.exists():
        df_t2y = load_and_resample(t2y_path, value_col = 'treasury_2y', method= 'last')
        data['treasury_2y'] = df_t2y
        print(f"✓ Treasury 2Y: {len(df_t2y)} meses" )
    else:
        print(f"  ✗ Archivo no encontrado: {t2y_path}")
        data['treasury_2y'] = None
    
    # 6. Treasury 10y (diario -> mensual, último valor)
    print("\n[6/9] Treasury Bonds 10 year rate...")
    t10y_path = DATA_RAW / "fred_treasury_10y.csv"
    if t10y_path.exists():
        df_t10y = load_and_resample(t10y_path, value_col = 'treasury_10y', method= 'last')
        data['treasury_10y'] = df_t10y
        print(f"✓ Treasury 2Y: {len(df_t10y)} meses" )
    else:
        print(f"  ✗ Archivo no encontrado: {t10y_path}")
        data['treasury_10y'] = None

    # 7. Spread BBB (diario -> mensual, último valor)
    print("\n[7/9] Spread BBB...")
    spread_path = DATA_RAW / "fred_spread_bbb.csv"
    if spread_path.exists():
        df_spread = load_and_resample(spread_path, value_col = 'spread_bbb', method = 'last')
        data['spread_bbb'] = df_spread
        print(f"  ✓ Spread BBB: {len(df_spread)} meses")
    else:
        print(f"  ✗ Archivo no encontrado: {spread_path}")
        data['spread_bbb'] = None

    # 8. PIB (trimestral → mantener trimestral por ahora)
    print("\n[8/9] PIB nominal")
    gdp_path = DATA_RAW / "fred_gdp_nominal.csv"
    if gdp_path.exists():
        df_gdp = pd.read_csv(gdp_path, parse_dates = ['date'])
    #Por ahora dejar así PIB nominal. En un futuro haremos interpolación.
        data['gdp'] = df_gdp
        print(f"  ✓ PIB: {len(df_gdp)} trimestres")
    else:
        print(f"  ✗ Archivo no encontrado: {gdp_path}")
        data['gdp'] = None

    # 9. Shiller CAPE (mensual, ya está en frecuencia correcta)
    print("\n[9/9] Shiller CAPE...")
    shiller_path = DATA_EXTERNAL / "shiller_cape.csv"
    if shiller_path.exists():
        try:
            #Intentar primero UTF-8
            df_shiller = pd.read_csv(shiller_path, parse_dates=['date'], encoding='utf-8')
            data['shiller'] = df_shiller
            print(f" ✓ Shiller CAPE: {len(df_shiller)} meses")
        except UnicodeDecodeError:
            # Fallback a latin-1 si UTF-8 falla
            df_shiller = pd.read_csv(shiller_path, parse_dates=['date'], encoding='latin-1')
            data['shiller'] = df_shiller
            print(f" ✓ Shiller CAPE: {len(df_shiller)} meses")
    else:
        print(f"  ✗ Archivo no encontrado: {shiller_path}")
        data['shiller'] = None

    return data

# Merge de Todas las Series
def merge_all_series(data):
    """
    Hace merge de todas las series en un solo DataFrame mensual
    
    Parameters
    ----------
    data : dict
        Diccionario con DataFrames por categoría
    
    Returns
    -------
    pd.DataFrame
        DataFrame mensual unificado
    """
    print("\n" + "="*70)
    print("COMBINANDO TODAS LAS SERIES")
    print("="*70)
    
    # S&P 500 como base
    if data['sp500'] is not None:
        df_monthly = data['sp500'].copy()
        print(f"  Base: S&P 500 ({len(df_monthly)} obs)")
    else:
        raise ValueError("S&P 500 es obligatorio como serie base")
    
    #Normaización de fechas: Conversión a "final de mes" para hacer el merge
    df_monthly['date'] = pd.to_datetime(df_monthly['date']).dt.to_period('M').dt.to_timestamp('M')
    print(f"[DEBUG] Fechas S&P normalizadas a fim de mes")
    print(f"[DEBUG] ejemplo fechas: {df_monthly['date'].head(3).tolist()}")

    #Normalización en todos lo DataFrames antes del merge
    for key in ['vix', 'fed_balance', 'ff_rate', 'treasury_2y', 'treasury_10y', 'spread_bbb']:
        if data[key] is None:
            data[key]['date'] = pd.to_datetime(data[key]['date']).dt.to_period('M').dt.to_timestamp('M')

    # Lista de series a mergear (orden de prioridad)
    merge_order = [
        'vix', 'fed_balance', 'ff_rate', 
        'treasury_2y', 'treasury_10y', 'spread_bbb'
    ]
    
    for key in merge_order:
        if data[key] is not None:
            df_monthly = pd.merge(
                df_monthly,
                data[key],
                on='date',
                how='left'
            )
            print(f"  + {key:15s} ({len(data[key])} obs)")
    
    # PIB
    if data['gdp'] is not None:
        gdp_df = data['gdp'].copy()
        
        # Verificar qué columna tiene los datos de GDP
        print(f"  [DEBUG GDP] Columnas en GDP: {gdp_df.columns.tolist()}")
        
        # Normalización de fechas a fin de mes
        gdp_df['date'] = pd.to_datetime(gdp_df['date']).dt.to_period('M').dt.to_timestamp('M')

        print(f" [DEBUG GDP] Columnas: {gdp_df.columns.tolist()}")
        print(f"[DEBUG GDP] Primeras fechas: {gdp_df['date'].head(3).tolist()}")
        
        # Renombrar la columna de GDP si es necesario
        gdp_col_name = None
        for col in gdp_df.columns:
            if col.lower().strip() in ['gdp', 'gdp_nominal']:
                gdp_col_name = col
                break
        
        if gdp_col_name and gdp_col_name != 'gdp_nominal':
            gdp_df = gdp_df.rename(columns={gdp_col_name: 'gdp_nominal'})
            print(f"  [DEBUG GDP] Renombrado '{gdp_col_name}' → 'gdp_nominal'")
        
        # Verificar que existe la columna
        if 'gdp_nominal' in gdp_df.columns:
            # Merge directo
            df_monthly = pd.merge(
                df_monthly,
                gdp_df[['date', 'gdp_nominal']],
                on='date',
                how='left'
            )
            
            # Interpolar linealmente
            df_monthly['gdp_nominal'] = df_monthly['gdp_nominal'].interpolate(method='linear')
            
            # Verificar
            gdp_not_null = df_monthly['gdp_nominal'].notna().sum()
            print(f"  + gdp_nominal   (interpolado: {gdp_not_null}/{len(df_monthly)} meses con datos)")
        else:
            print(f"  ✗ ERROR: No se encontró columna de GDP")
    
    # SHILLER: CORRECCIÓN SIMPLE
    if data['shiller'] is not None:
        shiller_df = data['shiller'].copy()
        
        print(f"  [DEBUG] Columnas Shiller: {shiller_df.columns.tolist()}")
        
        # Normalización fecha a fin de mes
        shiller_df['date'] = pd.to_datetime(shiller_df['date']).dt.to_period('M').dt.to_timestamp('M')

        print(f"[DEBUG] Columnas Shiller: {shiller_df.columns.tolist()}")
        print(f"[DEBUG] Primeras Filas Shiller: {shiller_df['date'].head(3).tolist()}")
              
        # Renombrar solo price y dividend (earnings y cape ya tienen nombres correctos)
        rename_dict = {}
        if 'price' in shiller_df.columns:
            rename_dict['price'] = 'shiller_price'
        if 'dividend' in shiller_df.columns:
            rename_dict['dividend'] = 'shiller_dividend'
        
        if rename_dict:
            shiller_df = shiller_df.rename(columns=rename_dict)
            print(f"  [DEBUG] Renombrado: {rename_dict}")
        
        # Merge
        df_monthly = pd.merge(
            df_monthly,
            shiller_df,
            on='date',
            how='left'
        )
        
        # Contar datos válidos
        if 'earnings' in df_monthly.columns:
            earnings_count = df_monthly['earnings'].notna().sum()
            cape_count = df_monthly['cape'].notna().sum() if 'cape' in df_monthly.columns else 0
            print(f"  + shiller_data  ({len(data['shiller'])} obs)")
            print(f"    - earnings: {earnings_count} valores válidos")
            print(f"    - cape: {cape_count} valores válidos")
        else:
            print(f"  + shiller_data  ({len(data['shiller'])} obs)")
    
    # Ordenar por fecha
    df_monthly = df_monthly.sort_values('date').reset_index(drop=True)
    
    print(f"\n  → DataFrame final: {len(df_monthly)} meses × {len(df_monthly.columns)} columnas")
    
    return df_monthly

# Cálculo de Transformaciones
def calculo_transformaciones(df):
    """
    Calcula transformaciones: logs, diferencias, rendimientos
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mensual con series en niveles
    
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas transformadas añadidas
    """
    print("\n" + "="*70)
    print("CALCULANDO TRANSFORMACIONES")
    print("="*70)

    df = df.copy()
    
# Logaritmos
    print("\n Logaritnos naturales:")
    if 'sp500' in df.columns:
        df['log_sp500'] = np.log(df['sp500'])
        print(" ✓ log_sp500")
    
    if 'fed_balance' in df.columns:
        df['log_balance'] = np.log(df['fed_balance'])
        print("    ✓ log_balance")
    
    if 'earnings' in df.columns:
        # Eliminar earnings negativos o cero (raros pero ocurren en crisis)
        df['earnings_clean'] = df['earnings'].replace(0, np.nan)
        df['earnings_clean'] = df['earnings_clean'].apply(lambda x: x if x > 0 else np.nan)
        df['log_earnings'] = np.log(df['earnings_clean'])
        print("    ✓ log_earnings")
    
    if 'gdp_nominal' in df.columns:
        df['log_gdp'] = np.log(df['gdp_nominal'])
        print("    ✓ log_gdp")

# Rendimientos y Crecimientos
    print("\n  Rendimientos y crecimientos (Δlog):")

    if 'log_sp500' in df.columns:
        df['ret_sp50'] = df['log_sp500'].diff()
        print("✓ ret_sp500 (rendimiento mensual S&P 500)")

    if 'log_balance' in df.columns:
        df['growth_balance'] = df['log_balance'].diff()
        print( "✓ growth_balance (crecimiento mensual balance Fed)")

#Diferencias simples, para valores ya en porcentajes o puntos basicos
    print ("\n Diferencias simples:")

    if 'vix' in df.columns:
        df['delta_vix'] = df['vix'].diff()
        print(" ✓ delta_vix")

    if 'ff_rate' in df.columns:
        df['delta_ff'] = df['ff_rate'].diff()
        print(" ✓ delta_ff")

    if 'spread_bbb' in df.columns:
        df['delta_spread'] = df['ff_rate'].diff()
        print(" ✓ delta_spread")
    
# Pendiente de la Curva (10y - 2y)
    print("\n  Variables derivadas:")

    if 'treasury_10y' in df.columns and 'treasury_2y' in df.columns:
        df['slope_curve'] = df['treasury_10y'] - df['treasury_2y']
        df['delta_slope'] = df['slope_curve'].diff()
        print(" ✓ slope_curve (10Y - 2Y)")
        print("    ✓ delta_slope")

#Información de Valores Faltantes
    print("\n" + "="*70)
    print("RESUMEN DE VALORES FALTANTES")
    print("="*70)

    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df) * 100).round(2)

    missing_df = pd.DataFrame({
        'Variable': missing_summary.index,
        'NaN count': missing_summary.values,
        'NaN %': missing_pct.values
    })

    #Mostrar solo variables con NaN > 0
    missing_df = missing_df[missing_df['NaN count'] > 0].sort_values('NaN count', ascending=False)

    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("  ✓ No hay valores faltantes")
    
    return df

# FUNCION PRINCIPAL
def main():
    """
    Pipeline completo: carga, merge, transformaciones
    """
    print("\n" + "="*70)
    print(" PIPELINE DE CREACIÓN DE DATASET MENSUAL")
    print("="*70)
    print(f"Periodo: {START_DATE} a {END_DATE}")
    print(f"Frecuencia: Mensual (fin de mes)")
    print("="*70)

    start_time = datetime.now()
    data = load_all_data()
    df_monthly = merge_all_series(data)
    df_monthly = calculo_transformaciones(df_monthly)
    df_monthly = df_monthly[
        (df_monthly['date'] >= pd.to_datetime(START_DATE)) &
        (df_monthly['date'] <= pd.to_datetime(END_DATE))   
    ]

    # Guardar CSV Procesado
    output_path = DATA_PROCESSED / "monthly_data.csv"
    df_monthly.to_csv(output_path, index = False)

    # Guardado también en Pickle
    output_pkl = DATA_PROCESSED / "monthly_data.pkl"
    df_monthly.to_pickle(output_pkl)

    # Resumen Final
    print("\n" + "="*70)
    print("DATASET MENSUAL CREADO EXITOSAMENTE")
    print("="*70)
    print(f"Observaciones:  {len(df_monthly)}")
    print(f"Variables:      {len(df_monthly.columns)}")
    print(f"Periodo:        {df_monthly['date'].min().strftime('%Y-%m')} a "
          f"{df_monthly['date'].max().strftime('%Y-%m')}")
    print(f"Tiempo total:   {(datetime.now() - start_time).total_seconds():.1f}s")
    print(f"\nArchivos guardados:")
    print(f"  - {output_path}")
    print(f"  - {output_pkl}")

    # Muestra de Primeras Filas
    print("\n" + "="*70)
    print("VISTA PREVIA (primeras 5 filas)")
    print("="*70)
    print(df_monthly.head().to_string())

    # Muestra de últimas filas
    print("\n" + "="*70)
    print("VISTA PREVIA (últimas 5 filas)")
    print("="*70)
    print(df_monthly.tail().to_string())

    print("\n✓ Pipeline completado\n")

# Ejecución
if __name__ == "__main__":
     main()