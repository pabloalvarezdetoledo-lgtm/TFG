"""
Agregación a frecuencia mensual y cálculo de transformaciones
Lee datos en frecuencia original y agrega a mensual con transformaciones

Autor: Pablo Álvarez de Toledo Rodríguez
Fecha: Enero 2026
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Añadir src/ al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import (
    DATA_RAW,
    DATA_PROCESSED,
    DATA_EXTERNAL,
    START_DATE,
    END_DATE,
    MONTHLY_FREQ
)


# =============================================================================
# FUNCIÓN: RESAMPLE A MENSUAL
# =============================================================================
def resample_to_monthly(df, date_col='date', method='last'):
    """
    Resamplea DataFrame a frecuencia mensual
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columna de fecha
    date_col : str
        Nombre de la columna de fechas
    method : str
        Método de agregación ('last', 'mean', 'sum')
    
    Returns
    -------
    pd.DataFrame
        DataFrame mensual con 'date' como columna
    """
     # Asegurar que date_col es datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
    # Set date as index
    df_indexed = df.set_index(date_col)
    
    # Forward fill para rellenar weekends/holidays
    # COMPATIBLE CON PANDAS 2.x
    df_filled = df_indexed.ffill()  # ← CAMBIO AQUÍ: ffill() en lugar de fillna(method='ffill')
    
    # Resample según método
    if method == 'last':
        df_monthly = df_filled.resample('ME').last()
    elif method == 'mean':
        df_monthly = df_filled.resample('ME').mean()
    elif method == 'sum':
        df_monthly = df_filled.resample('ME').sum()
    else:
        raise ValueError(f"Método '{method}' no reconocido")
    
    # Reset index para tener 'date' como columna
    df_monthly = df_monthly.reset_index()
    
    return df_monthly

# =============================================================================
# FUNCIÓN: LOAD AND RESAMPLE
# =============================================================================

def load_and_resample(filepath, value_col, method='last'):
    """
    Carga CSV y resamplea a mensual en un paso
    
    Parameters
    ----------
    filepath : Path
        Path al archivo CSV
    value_col : str
        Nombre de la columna de valores
    method : str
        Método de agregación ('last', 'mean', 'sum')
    
    Returns
    -------
    pd.DataFrame
        DataFrame mensual con columnas ['date', value_col]
    """
    # BUG 4 CORREGIDO: Indentación correcta
    df = pd.read_csv(filepath, parse_dates=['date'])
    
    # Seleccionar solo columnas relevantes
    df = df[['date', value_col]]
    
    # Resample
    df_monthly = resample_to_monthly(df, method=method)
    
    return df_monthly


# =============================================================================
# FUNCIÓN: LOAD ALL DATA
# =============================================================================

def load_all_data():
    """
    Carga todas las series y las resamplea a mensual
    
    Returns
    -------
    dict
        Diccionario con DataFrames mensuales
    """
    print("\n" + "="*70)
    print("CARGANDO Y RESAMPLING A MENSUAL")
    print("="*70)
    
    data = {}
    
    # -------------------------------------------------------------------------
    # 1. S&P 500 (diario → mensual, último valor)
    # -------------------------------------------------------------------------
    print("\n[1/9] S&P 500...")
    sp500_path = DATA_RAW / "yahoo_sp500.csv"
    if sp500_path.exists():
        data['sp500'] = load_and_resample(sp500_path, 'sp500', method='last')
        print(f"  ✓ S&P 500: {len(data['sp500'])} meses")
    else:
        print(f"  ✗ Archivo no encontrado: {sp500_path}")
        data['sp500'] = None
    
    # -------------------------------------------------------------------------
    # 2. VIX (diario → mensual, último valor)
    # -------------------------------------------------------------------------
    print("\n[2/9] VIX...")
    vix_path = DATA_RAW / "yahoo_vix.csv"
    if vix_path.exists():
        data['vix'] = load_and_resample(vix_path, 'vix', method='last')
        print(f"  ✓ VIX: {len(data['vix'])} meses")
    else:
        print(f"  ✗ Archivo no encontrado: {vix_path}")
        data['vix'] = None
    
    # -------------------------------------------------------------------------
    # 3. Balance de la Fed (semanal → mensual, último valor)
    # -------------------------------------------------------------------------
    print("\n[3/9] Balance Fed...")
    balance_path = DATA_RAW / "fred_fed_balance.csv"
    if balance_path.exists():
        data['fed_balance'] = load_and_resample(balance_path, 'fed_balance', method='last')
        print(f"  ✓ Balance Fed: {len(data['fed_balance'])} meses")
    else:
        print(f"  ✗ Archivo no encontrado: {balance_path}")
        data['fed_balance'] = None
    
    # -------------------------------------------------------------------------
    # 4. Fed Funds Rate (diario → mensual, último valor)
    # -------------------------------------------------------------------------
    print("\n[4/9] Fed Funds Rate...")
    ff_path = DATA_RAW / "fred_ff_rate.csv"
    if ff_path.exists():
        data['ff_rate'] = load_and_resample(ff_path, 'ff_rate', method='last')
        print(f"  ✓ Fed Funds: {len(data['ff_rate'])} meses")
    else:
        # BUG 5 CORREGIDO: Añadido print()
        print(f"  ✗ Archivo no encontrado: {ff_path}")
        data['ff_rate'] = None
    
    # -------------------------------------------------------------------------
    # 5. Treasury 2Y (diario → mensual, último valor)
    # -------------------------------------------------------------------------
    print("\n[5/9] Treasury 2Y...")
    t2y_path = DATA_RAW / "fred_treasury_2y.csv"
    if t2y_path.exists():
        data['treasury_2y'] = load_and_resample(t2y_path, 'treasury_2y', method='last')
        print(f"  ✓ Treasury 2Y: {len(data['treasury_2y'])} meses")
    else:
        print(f"  ✗ Archivo no encontrado: {t2y_path}")
        data['treasury_2y'] = None
    
    # -------------------------------------------------------------------------
    # 6. Treasury 10Y (diario → mensual, último valor)
    # -------------------------------------------------------------------------
    print("\n[6/9] Treasury 10Y...")
    t10y_path = DATA_RAW / "fred_treasury_10y.csv"
    if t10y_path.exists():
        data['treasury_10y'] = load_and_resample(t10y_path, 'treasury_10y', method='last')
        print(f"  ✓ Treasury 10Y: {len(data['treasury_10y'])} meses")
    else:
        print(f"  ✗ Archivo no encontrado: {t10y_path}")
        data['treasury_10y'] = None
    
    # -------------------------------------------------------------------------
    # 7. Spread BBB (diario → mensual, último valor)
    # -------------------------------------------------------------------------
    print("\n[7/9] Spread BBB...")
    spread_path = DATA_RAW / "fred_spread_bbb.csv"
    if spread_path.exists():
        data['spread_bbb'] = load_and_resample(spread_path, 'spread_bbb', method='last')
        print(f"  ✓ Spread BBB: {len(data['spread_bbb'])} meses")
    else:
        print(f"  ✗ Archivo no encontrado: {spread_path}")
        data['spread_bbb'] = None
    
    # -------------------------------------------------------------------------
    # 8. GDP (trimestral, NO resamplear todavía - interpolar en merge)
    # -------------------------------------------------------------------------
    print("\n[8/9] GDP...")
    gdp_path = DATA_RAW / "fred_gdp_nominal.csv"
    if gdp_path.exists():
        df_gdp = pd.read_csv(gdp_path, parse_dates=['date'])
        data['gdp_nominal'] = df_gdp[['date', 'gdp_nominal']]
        print(f"  ✓ GDP: {len(data['gdp_nominal'])} trimestres (interpolar después)")
    else:
        print(f"  ✗ Archivo no encontrado: {gdp_path}")
        data['gdp_nominal'] = None
    
    # -------------------------------------------------------------------------
    # 9. Shiller CAPE (ya mensual)
    # -------------------------------------------------------------------------
    print("\n[9/9] Shiller CAPE...")
    shiller_path = DATA_EXTERNAL / "shiller_cape.csv"
    if shiller_path.exists():
        df_shiller = pd.read_csv(shiller_path, parse_dates=['date'])
        data['shiller'] = df_shiller[['date', 'price', 'dividend', 'earnings', 'cape']]
        print(f"  ✓ Shiller: {len(data['shiller'])} meses")
    else:
        print(f"  ✗ Archivo no encontrado: {shiller_path}")
        data['shiller'] = None
    
    return data


# =============================================================================
# FUNCIÓN: MERGE ALL SERIES
# =============================================================================

def merge_all_series(data):
    """
    Hace merge de todas las series en un único DataFrame mensual
    
    Parameters
    ----------
    data : dict
        Diccionario con DataFrames de cada serie
    
    Returns
    -------
    pd.DataFrame
        DataFrame con todas las series alineadas mensualmente
    """
    print("\n" + "="*70)
    print("MERGE DE SERIES A FRECUENCIA MENSUAL")
    print("="*70)
    
    # BUG 2 CORREGIDO: Cambiar "is None" por "is not None"
    # Normalizar formato de fechas en todas las series
    for key in data.keys():
        if data[key] is not None:  # ← CORRECCIÓN AQUÍ
            data[key]['date'] = pd.to_datetime(data[key]['date'])
            # Asegurar que sea fin de mes
            data[key]['date'] = data[key]['date'] + pd.offsets.MonthEnd(0)
    
    # Base: S&P 500 (todas las demás series se alinean a estas fechas)
    if data['sp500'] is None:
        raise ValueError("S&P 500 es obligatorio como serie base")
    
    df_monthly = data['sp500'].copy()
    print(f"Base: S&P 500 ({len(df_monthly)} meses)")
    
    # Merge secuencial con left join (preserva fechas de SP500)
    
    # VIX
    if data['vix'] is not None:
        df_monthly = pd.merge(df_monthly, data['vix'], on='date', how='left')
        print(f"  + vix           ({len(data['vix'])} meses)")
    
    # Balance Fed
    if data['fed_balance'] is not None:
        df_monthly = pd.merge(df_monthly, data['fed_balance'], on='date', how='left')
        print(f"  + fed_balance   ({len(data['fed_balance'])} meses)")
    
    # Fed Funds
    if data['ff_rate'] is not None:
        df_monthly = pd.merge(df_monthly, data['ff_rate'], on='date', how='left')
        print(f"  + ff_rate       ({len(data['ff_rate'])} meses)")
    
    # Treasury 2Y
    if data['treasury_2y'] is not None:
        df_monthly = pd.merge(df_monthly, data['treasury_2y'], on='date', how='left')
        print(f"  + treasury_2y   ({len(data['treasury_2y'])} meses)")
    
    # Treasury 10Y
    if data['treasury_10y'] is not None:
        df_monthly = pd.merge(df_monthly, data['treasury_10y'], on='date', how='left')
        print(f"  + treasury_10y  ({len(data['treasury_10y'])} meses)")
    
    # Spread BBB
    if data['spread_bbb'] is not None:
        df_monthly = pd.merge(df_monthly, data['spread_bbb'], on='date', how='left')
        print(f"  + spread_bbb    ({len(data['spread_bbb'])} meses)")
    
    # GDP: merge con interpolación
    if data['gdp_nominal'] is not None:
        df_monthly = pd.merge(
            df_monthly,
            data['gdp_nominal'][['date', 'gdp_nominal']],
            on='date',
            how='left'
        )
        
        # Interpolar linealmente
        df_monthly['gdp_nominal'] = df_monthly['gdp_nominal'].interpolate(method='linear')
        print(f"  + gdp_nominal   (interpolado a {len(df_monthly)} meses)")
    
    # Shiller: merge de todas las columnas
    if data['shiller'] is not None:
        # Renombrar columnas para evitar conflictos
        shiller_cols = data['shiller'].copy()
        shiller_cols.columns = ['date', 'shiller_price', 'shiller_dividend', 'earnings', 'cape']
        
        df_monthly = pd.merge(df_monthly, shiller_cols, on='date', how='left')
        print(f"  + shiller (price, dividend, earnings, cape)")
    
    print(f"\nDataFrame final: {len(df_monthly)} meses × {len(df_monthly.columns)} variables")
    
    return df_monthly


# =============================================================================
# FUNCIÓN: CALCULATE TRANSFORMATIONS
# =============================================================================

def calculate_transformations(df):
    """
    Calcula transformaciones (logs, diferencias, ratios)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con series en niveles
    
    Returns
    -------
    pd.DataFrame
        DataFrame con transformaciones añadidas
    """
    print("\n" + "="*70)
    print("CALCULANDO TRANSFORMACIONES")
    print("="*70)
    
    # -------------------------------------------------------------------------
    # LOGARITMOS (para series que crecen exponencialmente)
    # -------------------------------------------------------------------------
    print("\n  Logaritmos:")
    
    # Log S&P 500
    if 'sp500' in df.columns:
        df['log_sp500'] = np.log(df['sp500'])
        print("    ✓ log_sp500")
    
    # Log Balance Fed
    if 'fed_balance' in df.columns:
        df['log_balance'] = np.log(df['fed_balance'])
        print("    ✓ log_balance")
    
    # Log Earnings (con manejo de negativos)
    if 'earnings' in df.columns:
        # Reemplazar 0 y negativos por NaN antes de log
        df['earnings_clean'] = df['earnings'].replace(0, np.nan)
        df['earnings_clean'] = df['earnings_clean'].apply(lambda x: x if x > 0 else np.nan)
        df['log_earnings'] = np.log(df['earnings_clean'])
        print("    ✓ log_earnings (negativos → NaN)")
    
    # Log GDP
    if 'gdp_nominal' in df.columns:
        df['log_gdp'] = np.log(df['gdp_nominal'])
        print("    ✓ log_gdp")
    
    # -------------------------------------------------------------------------
    # RENDIMIENTOS Y CRECIMIENTOS (diferencias de logs)
    # -------------------------------------------------------------------------
    print("\n  Rendimientos/Crecimientos:")
    
    # Rendimiento S&P 500
    if 'log_sp500' in df.columns:
        df['ret_sp500'] = df['log_sp500'].diff()
        print("    ✓ ret_sp500 (rendimiento mensual)")
    
    # Crecimiento Balance
    if 'log_balance' in df.columns:
        df['growth_balance'] = df['log_balance'].diff()
        print("    ✓ growth_balance (crecimiento mensual balance)")
    
    # -------------------------------------------------------------------------
    # DIFERENCIAS SIMPLES (para variables ya en % o basis points)
    # -------------------------------------------------------------------------
    print("\n  Diferencias simples:")
    
    # Cambio en VIX
    if 'vix' in df.columns:
        df['delta_vix'] = df['vix'].diff()
        print("    ✓ delta_vix")
    
    # Cambio en Fed Funds
    if 'ff_rate' in df.columns:
        df['delta_ff'] = df['ff_rate'].diff()
        print("    ✓ delta_ff")
    
    # Cambio en Spread BBB
    if 'spread_bbb' in df.columns:
        # BUG 3 CORREGIDO: Usar spread_bbb en lugar de ff_rate
        df['delta_spread'] = df['spread_bbb'].diff()  # ← CORRECCIÓN AQUÍ
        print("    ✓ delta_spread")
    
    # -------------------------------------------------------------------------
    # VARIABLES DERIVADAS
    # -------------------------------------------------------------------------
    print("\n  Variables derivadas:")
    
    # Pendiente de la curva (slope)
    if 'treasury_10y' in df.columns and 'treasury_2y' in df.columns:
        df['slope_curve'] = df['treasury_10y'] - df['treasury_2y']
        print("    ✓ slope_curve (10Y - 2Y)")
        
        # Cambio en pendiente
        df['delta_slope'] = df['slope_curve'].diff()
        print("    ✓ delta_slope")
    
    # -------------------------------------------------------------------------
    # RESUMEN DE MISSING VALUES
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("RESUMEN DE VALORES FALTANTES")
    print("="*70)
    
    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df) * 100).round(2)
    
    print(f"\n{'Variable':<20} {'Missing':<10} {'%':<10}")
    print("-"*40)
    for var in df.columns:
        if var != 'date':
            n_missing = missing_summary[var]
            pct_missing = missing_pct[var]
            if n_missing > 0:
                print(f"{var:<20} {n_missing:<10} {pct_missing:<10.2f}")
    
    return df


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Pipeline completo de agregación mensual
    """
    print("\n" + "="*70)
    print(" PIPELINE DE AGREGACIÓN A FRECUENCIA MENSUAL")
    print("="*70)
    print(f"Periodo objetivo: {START_DATE} a {END_DATE}")
    print("="*70)
    
    start_time = datetime.now()
    
    # Paso 1: Cargar datos
    data = load_all_data()
    
    # Paso 2: Merge
    df_monthly = merge_all_series(data)
    
    # Paso 3: Transformaciones
    df_monthly = calculate_transformations(df_monthly)
    
    # Paso 4: Filtrar por rango de fechas
    print("\n" + "="*70)
    print("FILTRADO POR RANGO DE FECHAS")
    print("="*70)
    
    df_monthly = df_monthly[
        (df_monthly['date'] >= pd.to_datetime(START_DATE)) &
        (df_monthly['date'] <= pd.to_datetime(END_DATE))
    ]
    
    print(f"Observaciones en rango: {len(df_monthly)}")
    print(f"Periodo final: {df_monthly['date'].min().strftime('%Y-%m')} a "
          f"{df_monthly['date'].max().strftime('%Y-%m')}")
    
    # Paso 5: Guardar
    print("\n" + "="*70)
    print("GUARDANDO RESULTADOS")
    print("="*70)
    
    # CSV (human-readable)
    output_csv = DATA_PROCESSED / "monthly_data.csv"
    df_monthly.to_csv(output_csv, index=False)
    size_csv = output_csv.stat().st_size / 1024
    print(f"  ✓ CSV guardado: {output_csv} ({size_csv:.1f} KB)")
    
    # Pickle (Python-optimized)
    output_pkl = DATA_PROCESSED / "monthly_data.pkl"
    df_monthly.to_pickle(output_pkl)
    size_pkl = output_pkl.stat().st_size / 1024
    print(f"  ✓ Pickle guardado: {output_pkl} ({size_pkl:.1f} KB)")
    
    # Resumen
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*70)
    print("PIPELINE COMPLETADO")
    print("="*70)
    print(f"Tiempo total: {elapsed:.1f}s")
    print(f"Dataset final: {len(df_monthly)} meses × {len(df_monthly.columns)} variables")
    print("\n✓ Datos listos para análisis\n")


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    main()