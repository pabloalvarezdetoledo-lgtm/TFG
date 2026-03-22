"""
VECM con Net Liquidity - Múltiples especificaciones de lags
Contrasta sensibilidad del resultado a la selección de lags

Autor: Pablo Álvarez de Toledo Rodríguez
Fecha: Marzo 2026
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, coint_johansen
from datetime import datetime
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import DATA_PROCESSED, RESULTS_TABLES

warnings.filterwarnings('ignore')


def run_johansen_test(data, endog_vars, lags, det_order=1, name=""):
    """
    Ejecuta test de Johansen con especificación dada
    
    Returns
    -------
    dict con resultados
    """
    print("\n" + "="*70)
    print(f"TEST DE JOHANSEN - {name}")
    print("="*70)
    print(f"Lags en diferencias: {lags - 1}")
    print(f"det_order: {det_order}")
    
    try:
        joh = coint_johansen(
            data[endog_vars],
            det_order=det_order,
            k_ar_diff=lags - 1
        )
        
        trace_stat = joh.lr1
        crit_vals = joh.cvt
        n_vars = len(endog_vars)
        
        # Test de traza
        print("\n" + "-"*70)
        print("TEST DE TRAZA")
        print("-"*70)
        print(f"{'H0: rank ≤':<15} {'Estadístico':<15} {'10%':<12} {'5%':<12} {'1%':<12}")
        print("-"*70)
        
        for i in range(n_vars):
            stat = trace_stat[i]
            cv_90 = crit_vals[i, 0]
            cv_95 = crit_vals[i, 1]
            cv_99 = crit_vals[i, 2]
            
            decision = "**" if stat > cv_95 else "*" if stat > cv_90 else ""
            print(f"{i:<15} {stat:<15.3f} {cv_90:<12.3f} {cv_95:<12.3f} {cv_99:<12.3f} {decision}")
        print("-"*70)
        
        # Determinar rango
        rank_5 = 0
        rank_10 = 0
        
        for i in range(n_vars):
            if trace_stat[i] > crit_vals[i, 1]:  # 5%
                rank_5 = i + 1
            if trace_stat[i] > crit_vals[i, 0]:  # 10%
                rank_10 = i + 1
        
        print(f"\n► Rango cointegración (5%):  {rank_5}")
        print(f"► Rango cointegración (10%): {rank_10}")
        
        if rank_5 >= 1:
            print(f"  ✓ Cointegración detectada al 5% (rank = {rank_5})")
        elif rank_10 >= 1:
            print(f"  ⚠ Cointegración solo al 10% (rank = {rank_10})")
        else:
            print("  ✗ No cointegración")
        
        return {
            'lags': lags,
            'name': name,
            'trace_stat_0': trace_stat[0],
            'crit_5': crit_vals[0, 1],
            'crit_10': crit_vals[0, 0],
            'rank_5': rank_5,
            'rank_10': rank_10,
            'joh_result': joh
        }
        
    except Exception as e:
        print(f"\n✗ Error en Johansen: {str(e)}")
        return None


def estimate_vecm_if_coint(data, endog_vars, lags, rank, name=""):
    """
    Estima VECM si hay cointegración
    """
    if rank == 0:
        print(f"\n⚠ No se estima VECM para {name} (rank = 0)")
        return None
    
    print("\n" + "="*70)
    print(f"ESTIMACIÓN VECM - {name}")
    print("="*70)
    
    try:
        model = VECM(
            endog=data[endog_vars],
            k_ar_diff=lags - 1,
            coint_rank=rank,
            deterministic='ci'
        )
        
        results = model.fit()
        
        # Mostrar summary completo para ver p-values
        print("\n" + "="*70)
        print("SUMMARY COMPLETO (para verificar significancia)")
        print("="*70)
        try:
            print(results.summary())
        except:
            print("⚠ No se puede mostrar summary completo")
        
        # Guardar beta y alpha con más detalle
        df_beta = pd.DataFrame(
            beta,
            index=endog_vars,
            columns=[f'β_{i+1}' for i in range(beta.shape[1])]
        )

        df_alpha = pd.DataFrame(
            alpha,
            index=endog_vars,
            columns= [f'α_{i+1}' for i in range(alpha.shape[1])]
        )

        # Guardar
        name_clean = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
        RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
        df_beta.to_csv(RESULTS_TABLES / f"vecm_beta_{name_clean}.csv")
        df_alpha.to_csv(RESULTS_TABLES / f"vecm_alpha_{name_clean}.csv")

        print(f"\n✓ Parámetros guardados:")
        print(f"  Beta:  vecm_beta_{name_clean}.csv")
        print(f"  Alpha: vecm_alpha_{name_clean}.csv")

        
        # Extraer beta
        beta = results.beta
        alpha = results.alpha
        
        print("\nVector de cointegración β (normalizado):")
        for i, var in enumerate(endog_vars):
            print(f"  {var:20s}: {beta[i, 0]:8.4f}")
        
        # Coeficiente de liquidez
        if 'log_net_liquidity' in endog_vars:
            idx = endog_vars.index('log_net_liquidity')
            beta_liq = beta[idx, 0]
            
            print(f"\n► β_net_liquidity = {beta_liq:.4f}")
            
            # Buscar p-value en summary
            try:
                summary_str = str(results.summary())
                
                # Interpretación
                if abs(beta_liq) > 0.3:
                    print(f"  Elasticidad implícita: {-beta_liq:.4f}")
                    print(f"  ✓ Coeficiente económicamente significativo")
                else:
                    print(f"  ⚠ Coeficiente pequeño (|β| < 0.3)")
                    
            except:
                pass
        
        return {
            'results': results,
            'beta': beta,
            'alpha': alpha
        }
        
    except Exception as e:
        print(f"\n✗ Error estimando VECM: {str(e)}")
        return None


def main():
    """Pipeline completo"""
    print("\n" + "="*70)
    print(" VECM NET LIQUIDITY - ANÁLISIS DE SENSIBILIDAD A LAGS")
    print("="*70)
    
    start_time = datetime.now()
    
    # Cargar datos
    print("\nCargando datos...")
    df = pd.read_csv(DATA_PROCESSED / "monthly_data.csv", parse_dates=['date'])
    
    if 'log_net_liquidity' not in df.columns:
        print("\n❌ ERROR: log_net_liquidity no encontrado")
        return
    
    endog_vars = ['log_sp500', 'log_earnings', 'log_net_liquidity']
    data = df[endog_vars].dropna()
    
    print(f"✓ Datos: {len(data)} observaciones")
    print(f"  Variables: {endog_vars}")
    
    # Primero: criterios de información
    print("\n" + "="*70)
    print("SELECCIÓN AUTOMÁTICA DE LAGS")
    print("="*70)
    
    lag_order_results = select_order(data, maxlags=12, deterministic='ci')
    
    print("\nCriterios de información:")
    print(f"  AIC:  {lag_order_results.aic} lags")
    print(f"  BIC:  {lag_order_results.bic} lags")
    print(f"  HQIC: {lag_order_results.hqic} lags")
    
    # Tests con 3 especificaciones
    specs = [
        {'lags': 1, 'name': 'BIC (1 lag)', 'criterion': 'BIC'},
        {'lags': 2, 'name': 'HQIC (2 lags)', 'criterion': 'HQIC'},
        {'lags': 12, 'name': 'AIC (12 lags)', 'criterion': 'AIC'}
    ]
    
    results_summary = []
    vecm_results = {}
    
    for spec in specs:
        result = run_johansen_test(
            data=data,
            endog_vars=endog_vars,
            lags=spec['lags'],
            det_order=1,
            name=spec['name']
        )
        
        if result:
            results_summary.append({
                'Especificación': spec['name'],
                'Criterio': spec['criterion'],
                'Lags': spec['lags'],
                'k_ar_diff': spec['lags'] - 1,
                'Estadístico': result['trace_stat_0'],
                'Crítico_5%': result['crit_5'],
                'Crítico_10%': result['crit_10'],
                'Rechaza_5%': '✓' if result['rank_5'] >= 1 else '✗',
                'Rechaza_10%': '✓' if result['rank_10'] >= 1 else '✗',
                'Rank_5%': result['rank_5'],
                'Rank_10%': result['rank_10']
            })
            
            # Estimar VECM si hay cointegración
            if result['rank_5'] >= 1:
                vecm_est = estimate_vecm_if_coint(
                    data=data,
                    endog_vars=endog_vars,
                    lags=spec['lags'],
                    rank=result['rank_5'],
                    name=spec['name']
                )
                
                if vecm_est:
                    vecm_results[spec['name']] = vecm_est
    
    # Tabla comparativa
    print("\n" + "="*70)
    print("TABLA COMPARATIVA - SENSIBILIDAD A LAGS")
    print("="*70)
    
    df_summary = pd.DataFrame(results_summary)
    print("\n" + df_summary.to_string(index=False))
    
    # Guardar
    output_path = RESULTS_TABLES / "johansen_lags_comparison_net_liquidity.csv"
    df_summary.to_csv(output_path, index=False)
    print(f"\n✓ Tabla guardada: {output_path}")
    
    # Conclusión
    print("\n" + "="*70)
    print("CONCLUSIÓN METODOLÓGICA")
    print("="*70)
    
    n_coint_5 = sum([1 for r in results_summary if r['Rechaza_5%'] == '✓'])
    n_coint_10 = sum([1 for r in results_summary if r['Rechaza_10%'] == '✓'])
    
    if n_coint_5 >= 1:
        print(f"\n✓ Cointegración detectada al 5% en {n_coint_5}/3 especificaciones")
        print("  → H2 CONFIRMADA con Net Liquidity")
        print("  → Seleccionar especificación por criterio teórico (parsimonia)")
    elif n_coint_10 >= 1:
        print(f"\n⚠ Cointegración solo al 10% en {n_coint_10}/3 especificaciones")
        print("  → Evidencia débil de relación de largo plazo")
        print("  → Proceder con análisis de corto plazo (VAR)")
    else:
        print("\n✗ No cointegración en ninguna especificación")
        print("  → H2 REFUTADA con Net Liquidity")
        print("  → Relación es de corto plazo")
    
    # Recomendación
    print("\n" + "-"*70)
    print("RECOMENDACIÓN:")
    print("-"*70)
    
    best_spec = df_summary.loc[df_summary['Lags'].idxmin()]
    
    if best_spec['Rechaza_5%'] == '✓':
        print(f"\nEspecificación recomendada: {best_spec['Especificación']}")
        print(f"  Razón: Parsimonia (menor lags) + cointegración al 5%")
    else:
        print("\nNo hay especificación con cointegración robusta")
        print("Proceder con VAR en diferencias (H3)")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱ Tiempo total: {elapsed:.1f}s")
    print("\n✓ Análisis completado\n")


if __name__ == "__main__":
    main()
    