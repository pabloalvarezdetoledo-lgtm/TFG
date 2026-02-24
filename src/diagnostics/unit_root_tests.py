"""
Tests de raíz unitaria para verificar orden de integración
Implementa tests ADF (Augmented Dickey-Fuller) y KPSS

Autor: Pablo Álvarez de Toledo Rodríguez
Fecha: Enero 2026
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from datetime import datetime
import warnings

# Añadir src/ al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import (
    DATA_PROCESSED,
    RESULTS_TABLES
)

# Suprimir warnings de convergencia (KPSS puede dar warnings en algunas series)
warnings.filterwarnings('ignore')


# FUNCIÓN: TEST ADF (AUGMENTED DICKEY-FULLER)

def adf_test(series, maxlag=None, regression='c', autolag='AIC'):
    """
    Realiza test ADF (Augmented Dickey-Fuller) para raíz unitaria
    
    Parameters
    ----------
    series : pd.Series
        Serie temporal a testear
    maxlag : int or None
        Número máximo de lags a considerar
    regression : str
        Tipo de regresión:
        - 'c': constante (default)
        - 'ct': constante + tendencia
        - 'ctt': constante + tendencia + tendencia cuadrática
        - 'n': sin constante ni tendencia
    autolag : str
        Criterio para selección de lags: 'AIC', 'BIC', 't-stat', None
    
    Returns
    -------
    dict
        Diccionario con estadísticos del test
        
    Notes
    -----
    Test ADF (Dickey-Fuller aumentado):
    
    Hipótesis nula H0: Serie tiene raíz unitaria (no estacionaria)
    Hipótesis alternativa H1: Serie es estacionaria
    
    Ecuación del test (con constante):
    Δy_t = α + β·t + γ·y_{t-1} + Σ δ_i·Δy_{t-i} + ε_t
    
    Se testea: H0: γ = 0 (raíz unitaria)
    
    Si p-value < 0.05: Rechazamos H0 → Serie es estacionaria
    Si p-value > 0.05: No rechazamos H0 → Serie tiene raíz unitaria
    
    Regla práctica:
    - Variables en NIVELES: Esperamos NO rechazar H0 (son I(1))
    - Variables en DIFERENCIAS: Esperamos rechazar H0 (son I(0))
    
    Referencias
    ----------
    - Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators 
      for autoregressive time series with a unit root. Journal of the 
      American Statistical Association, 74(366a), 427-431.
    - Said, S. E., & Dickey, D. A. (1984). Testing for unit roots in 
      autoregressive-moving average models of unknown order. Biometrika, 
      71(3), 599-607.
    """
    # Eliminar NaN
    series_clean = series.dropna()
    
    # Realizar test ADF
    result = adfuller(
        series_clean,
        maxlag=maxlag,
        regression=regression,
        autolag=autolag
    )
    
    # Extraer resultados
    adf_stat = result[0]
    p_value = result[1]
    n_lags = result[2]
    n_obs = result[3]
    critical_values = result[4]
    ic_best = result[5]
    
    # Decisión estadística
    stationary = p_value < 0.05
    
    return {
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'lags_used': n_lags,
        'n_observations': n_obs,
        'critical_value_1%': critical_values['1%'],
        'critical_value_5%': critical_values['5%'],
        'critical_value_10%': critical_values['10%'],
        'ic_best': ic_best,
        'stationary': stationary,
        'regression_type': regression
    }


# FUNCIÓN: TEST KPSS

def kpss_test(series, regression='c', nlags='auto'):
    """
    Realiza test KPSS (Kwiatkowski-Phillips-Schmidt-Shin) para estacionariedad
    
    Parameters
    ----------
    series : pd.Series
        Serie temporal a testear
    regression : str
        Tipo de regresión:
        - 'c': constante (default) - testea estacionariedad alrededor de nivel
        - 'ct': constante + tendencia - testea estacionariedad alrededor de tendencia
    nlags : str or int
        Número de lags para calcular estadístico:
        - 'auto': Usa fórmula de Schwert
        - 'legacy': Usa int(12 * (n/100)^(1/4))
        - int: Número específico de lags
    
    Returns
    -------
    dict
        Diccionario con estadísticos del test
        
    Notes
    -----
    Test KPSS (Kwiatkowski-Phillips-Schmidt-Shin):
    
    ¡ATENCIÓN! Este test tiene hipótesis INVERTIDA respecto a ADF:
    
    Hipótesis nula H0: Serie es estacionaria
    Hipótesis alternativa H1: Serie tiene raíz unitaria
    
    Si p-value < 0.05: Rechazamos H0 → Serie NO es estacionaria (tiene raíz unitaria)
    Si p-value > 0.05: No rechazamos H0 → Serie es estacionaria
    
    ¿Por qué usar KPSS además de ADF?
    
    Los tests tienen hipótesis complementarias:
    - ADF: H0 = raíz unitaria
    - KPSS: H0 = estacionaria
    
    Interpretación conjunta:
    
    | ADF rechaza H0 | KPSS rechaza H0 | Conclusión              |
    |----------------|-----------------|-------------------------|
    | Sí             | No              | Estacionaria (clara)    |
    | No             | Sí              | Raíz unitaria (clara)   |
    | No             | No              | Inconcluso (raro)       |
    | Sí             | Sí              | Ambiguo (KPSS dominante)|
    
    Referencias
    ----------
    - Kwiatkowski, D., Phillips, P. C., Schmidt, P., & Shin, Y. (1992). 
      Testing the null hypothesis of stationarity against the alternative 
      of a unit root. Journal of Econometrics, 54(1-3), 159-178.
    """
    # Eliminar NaN
    series_clean = series.dropna()
    
    # Realizar test KPSS
    result = kpss(
        series_clean,
        regression=regression,
        nlags=nlags
    )
    
    # Extraer resultados
    kpss_stat = result[0]
    p_value = result[1]
    n_lags = result[2]
    critical_values = result[3]
    
    # Decisión estadística (invertida respecto a ADF)
    stationary = p_value > 0.05  # No rechazamos H0 → estacionaria
    
    return {
        'kpss_statistic': kpss_stat,
        'p_value': p_value,
        'lags_used': n_lags,
        'critical_value_1%': critical_values['1%'],
        'critical_value_2.5%': critical_values['2.5%'],
        'critical_value_5%': critical_values['5%'],
        'critical_value_10%': critical_values['10%'],
        'stationary': stationary,
        'regression_type': regression
    }


# FUNCIÓN: TEST COMPLETO PARA UNA VARIABLE

def comprehensive_unit_root_test(series, var_name):
    """
    Realiza batería completa de tests para una variable
    
    Parameters
    ----------
    series : pd.Series
        Serie temporal a testear
    var_name : str
        Nombre de la variable
    
    Returns
    -------
    dict
        Diccionario con todos los resultados
        
    Notes
    -----
    Estrategia de testing:
    
    1. Test en NIVELES (con constante):
       - Si serie tiene tendencia determinística → usar 'ct'
       - Si no tiene tendencia → usar 'c'
       
    2. Test en DIFERENCIAS (con constante):
       - Diferencias no deberían tener tendencia
       - Usar 'c'
    
    Para variables financieras (log_sp500, log_balance):
    - Niveles: Probablemente I(1) → ADF no rechaza, KPSS rechaza
    - Diferencias: Probablemente I(0) → ADF rechaza, KPSS no rechaza
    
    Esto es lo que esperamos para poder usar VECM.
    """
    results = {
        'variable': var_name,
        'n_obs': len(series.dropna())
    }
    
   # TESTS EN NIVELES
    
    # ADF en niveles (con constante)
    adf_level = adf_test(series, regression='c', autolag='AIC')
    results['adf_level_stat'] = adf_level['adf_statistic']
    results['adf_level_pval'] = adf_level['p_value']
    results['adf_level_lags'] = adf_level['lags_used']
    results['adf_level_stationary'] = adf_level['stationary']
    
    # KPSS en niveles (con constante)
    kpss_level = kpss_test(series, regression='c', nlags='auto')
    results['kpss_level_stat'] = kpss_level['kpss_statistic']
    results['kpss_level_pval'] = kpss_level['p_value']
    results['kpss_level_lags'] = kpss_level['lags_used']
    results['kpss_level_stationary'] = kpss_level['stationary']
    
    # TESTS EN DIFERENCIAS
    
    # Calcular primera diferencia
    series_diff = series.diff().dropna()
    
    # ADF en diferencias
    adf_diff = adf_test(series_diff, regression='c', autolag='AIC')
    results['adf_diff_stat'] = adf_diff['adf_statistic']
    results['adf_diff_pval'] = adf_diff['p_value']
    results['adf_diff_lags'] = adf_diff['lags_used']
    results['adf_diff_stationary'] = adf_diff['stationary']
    
    # KPSS en diferencias
    kpss_diff = kpss_test(series_diff, regression='c', nlags='auto')
    results['kpss_diff_stat'] = kpss_diff['kpss_statistic']
    results['kpss_diff_pval'] = kpss_diff['p_value']
    results['kpss_diff_lags'] = kpss_diff['lags_used']
    results['kpss_diff_stationary'] = kpss_diff['stationary']
    
    # CONCLUSIÓN SOBRE ORDEN DE INTEGRACIÓN
    
    # Lógica de decisión basada en ambos tests
    level_nonstationary = (not adf_level['stationary']) or (not kpss_level['stationary'])
    diff_stationary = adf_diff['stationary'] and kpss_diff['stationary']
    
    if level_nonstationary and diff_stationary:
        integration_order = 'I(1)'
        conclusion = 'Serie integrada de orden 1 (apta para VECM)'
    elif adf_level['stationary'] and kpss_level['stationary']:
        integration_order = 'I(0)'
        conclusion = 'Serie estacionaria en niveles'
    elif not diff_stationary:
        integration_order = 'I(2)?'
        conclusion = 'Posiblemente integrada de orden 2 (verificar)'
    else:
        integration_order = 'Ambiguo'
        conclusion = 'Resultados contradictorios entre tests'
    
    results['integration_order'] = integration_order
    results['conclusion'] = conclusion
    
    return results


# FUNCIÓN: TESTEAR MÚLTIPLES VARIABLES

def test_multiple_series(df, variables):
    """
    Realiza tests de raíz unitaria para múltiples variables
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos
    variables : list
        Lista de nombres de columnas a testear
    
    Returns
    -------
    pd.DataFrame
        DataFrame con resultados para todas las variables
    """
    print("\n" + "="*70)
    print("TESTS DE RAÍZ UNITARIA (ADF Y KPSS)")
    print("="*70)
    print("\nTesteando orden de integración de las series...")
    print("H0 en ADF: Serie tiene raíz unitaria (no estacionaria)")
    print("H0 en KPSS: Serie es estacionaria")
    print("\nEsperamos para VECM: Series I(1) en niveles, I(0) en diferencias")
    print("="*70)
    
    results_list = []
    
    for i, var in enumerate(variables, 1):
        print(f"\n[{i}/{len(variables)}] Testeando: {var}")
        
        if var not in df.columns:
            print(f"  ✗ Variable '{var}' no encontrada en DataFrame")
            continue
        
        series = df[var]
        
        # Verificar si hay suficientes datos
        n_valid = series.notna().sum()
        if n_valid < 30:
            print(f"  ✗ Insuficientes observaciones válidas: {n_valid} < 30")
            continue
        
        # Realizar tests
        result = comprehensive_unit_root_test(series, var)
        results_list.append(result)
        
        # Imprimir resumen
        print(f"  Observaciones: {result['n_obs']}")
        print(f"  ADF (niveles):  stat={result['adf_level_stat']:.3f}, "
              f"p-val={result['adf_level_pval']:.4f}, "
              f"estac={'Sí' if result['adf_level_stationary'] else 'No'}")
        print(f"  KPSS (niveles): stat={result['kpss_level_stat']:.3f}, "
              f"p-val={result['kpss_level_pval']:.4f}, "
              f"estac={'Sí' if result['kpss_level_stationary'] else 'No'}")
        print(f"  ADF (diffs):    stat={result['adf_diff_stat']:.3f}, "
              f"p-val={result['adf_diff_pval']:.4f}, "
              f"estac={'Sí' if result['adf_diff_stationary'] else 'No'}")
        print(f"  KPSS (diffs):   stat={result['kpss_diff_stat']:.3f}, "
              f"p-val={result['kpss_diff_pval']:.4f}, "
              f"estac={'Sí' if result['kpss_diff_stationary'] else 'No'}")
        print(f"  ► Conclusión: {result['integration_order']} - {result['conclusion']}")
    
    # Convertir a DataFrame
    df_results = pd.DataFrame(results_list)
    
    return df_results


# FUNCIÓN: CREAR TABLA PARA TFG
def create_publication_table(df_results):
    """
    Crea tabla formateada para incluir en el TFG
    
    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame con resultados de tests
    
    Returns
    -------
    pd.DataFrame
        Tabla formateada lista para LaTeX o Word
    """
    # Seleccionar columnas relevantes y crear copia
    table = pd.DataFrame({
        'Variable': df_results['variable'],
        'ADF_niveles_stat': df_results['adf_level_stat'],
        'ADF_niveles_pval': df_results['adf_level_pval'],
        'ADF_diffs_stat': df_results['adf_diff_stat'],
        'ADF_diffs_pval': df_results['adf_diff_pval'],
        'Orden': df_results['integration_order']
    })
    
    # Formatear estadísticos ADF
    table['ADF (niveles)'] = table['ADF_niveles_stat'].apply(lambda x: f"{x:.3f}")
    table['ADF (diffs)'] = table['ADF_diffs_stat'].apply(lambda x: f"{x:.3f}")
    
    # Formatear p-values con lógica especial para valores muy pequeños
    def format_pvalue(x):
        if pd.isna(x):
            return 'NA'
        elif x < 0.0001:
            return '<0.0001'
        else:
            return f"{x:.4f}"
    
    table['p-value (niveles)'] = table['ADF_niveles_pval'].apply(format_pvalue)
    table['p-value (diffs)'] = table['ADF_diffs_pval'].apply(format_pvalue)
    
    # Seleccionar solo columnas finales
    table_final = table[[
        'Variable',
        'ADF (niveles)',
        'p-value (niveles)',
        'ADF (diffs)',
        'p-value (diffs)',
        'Orden'
    ]]
    
    return table_final

# FUNCIÓN PRINCIPAL
def main():
    """
    Pipeline completo de tests de raíz unitaria
    """
    print("\n" + "="*70)
    print(" PIPELINE DE TESTS DE RAÍZ UNITARIA")
    print("="*70)
    
    start_time = datetime.now()
    
    # Cargar datos
    print("\nCargando datos mensuales...")
    data_path = DATA_PROCESSED / "monthly_data.csv"
    df = pd.read_csv(data_path, parse_dates=['date'])
    print(f"  ✓ Datos cargados: {len(df)} observaciones")
    
    # Variables a testear (series clave para VECM y análisis)
    variables_to_test = [
        # Variables en niveles para VECM
        'log_sp500',       # Log S&P 500
        'log_balance',     # Log Balance Fed
        'log_earnings',    # Log Earnings
        
        # Variables adicionales
        'vix',             # VIX
        'ff_rate',         # Fed Funds Rate
        'treasury_10y',    # Treasury 10Y
        'treasury_2y',     # Treasury 2Y
        'spread_bbb',      # Spread BBB
        'slope_curve',     # Pendiente curva (10Y - 2Y)
        
        # Variables ya en diferencias (deberían ser I(0))
        'ret_sp500',       # Rendimiento S&P 500
        'growth_balance',  # Crecimiento balance
        'delta_vix',       # Cambio en VIX
        'delta_spread'     # Cambio en spread
    ]
    
    # Realizar tests
    df_results = test_multiple_series(df, variables_to_test)
    
    # Guardar resultados completos
    output_full = RESULTS_TABLES / "unit_root_tests_full.csv"
    df_results.to_csv(output_full, index=False)
    print(f"\n✓ Resultados completos guardados: {output_full}")
    
    # Crear tabla para publicación
    table_pub = create_publication_table(df_results)
    output_pub = RESULTS_TABLES / "unit_root_tests_publication.csv"
    table_pub.to_csv(output_pub, index=False)
    print(f"✓ Tabla para publicación guardada: {output_pub}")
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS")
    print("="*70)
    
    # Contar por orden de integración
    integration_counts = df_results['integration_order'].value_counts()
    print("\nDistribución de órdenes de integración:")
    for order, count in integration_counts.items():
        print(f"  {order}: {count} variables")
    
    # Variables I(1) (aptas para VECM)
    i1_vars = df_results[df_results['integration_order'] == 'I(1)']['variable'].tolist()
    print(f"\nVariables I(1) aptas para VECM:")
    for var in i1_vars:
        print(f"  - {var}")
    
    # Variables I(0) (ya estacionarias)
    i0_vars = df_results[df_results['integration_order'] == 'I(0)']['variable'].tolist()
    if i0_vars:
        print(f"\nVariables I(0) (estacionarias en niveles):")
        for var in i0_vars:
            print(f"  - {var}")
    
    # Advertencias
    problematic = df_results[
        ~df_results['integration_order'].isin(['I(0)', 'I(1)'])
    ]['variable'].tolist()
    
    if problematic:
        print(f"\n⚠ ADVERTENCIA: Variables con resultados ambiguos:")
        for var in problematic:
            conclusion = df_results[df_results['variable'] == var]['conclusion'].values[0]
            print(f"  - {var}: {conclusion}")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱ Tiempo total: {elapsed:.1f}s")
    print("\n✓ Pipeline completado\n")


# EJECUCIÓN
if __name__ == "__main__":
    main()