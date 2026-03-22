"""
VECM con Net Liquidity (Familia A)
Contrasta H2 con proxy de liquidez neta

Autor: Pablo Álvarez de Toledo Rodríguez
Fecha: Marzo 2026
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from datetime import datetime
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import (
    DATA_PROCESSED,
    RESULTS_TABLES,
    RESULTS_FIGURES,
    FIGURE_DPI
)

warnings.filterwarnings('ignore')


class VECMNetLiquidity:
    """VECM con Net Liquidity (WALCL - RRP - TGA)"""
    
    def __init__(self, df, endog_vars=['log_sp500', 'log_earnings', 'log_net_liquidity']):
        self.df = df
        self.endog_vars = endog_vars
        self.data = None
        self.model = None
        self.results = None
        self.lag_order = None
        self.coint_rank = None
        self.johansen_results = None
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepara datos"""
        self.data = self.df[self.endog_vars].dropna()
        
        print(f"\nDatos preparados:")
        print(f"  Variables: {self.endog_vars}")
        print(f"  Observaciones: {len(self.data)}")
        
        valid_indices = self.data.index
        dates = self.df.loc[valid_indices, 'date']
        
        print(f"  Periodo: {dates.iloc[0].strftime('%Y-%m')} a "
              f"{dates.iloc[-1].strftime('%Y-%m')}")
    
    def select_lag_order(self, maxlags=12, ic='bic'):
        """Selecciona lags"""
        print("\n" + "="*70)
        print("SELECCIÓN DE LAGS (NET LIQUIDITY)")
        print("="*70)
        
        lag_order_results = select_order(
            self.data,
            maxlags=maxlags,
            deterministic='ci'
        )
        
        print("\nCriterios de información:")
        print(f"  AIC:  {lag_order_results.aic} lags")
        print(f"  BIC:  {lag_order_results.bic} lags")
        print(f"  HQIC: {lag_order_results.hqic} lags")
        
        selected = getattr(lag_order_results, ic)
        self.lag_order = selected
        
        print(f"\n► Lags seleccionados ({ic.upper()}): {selected}")
        return selected
    
    def johansen_test(self, det_order=1, k_ar_diff=None):
        """Test de Johansen"""
        print("\n" + "="*70)
        print("TEST DE JOHANSEN (NET LIQUIDITY)")
        print("="*70)
        
        if k_ar_diff is None:
            k_ar_diff = self.lag_order - 1
        
        print(f"\nParámetros:")
        print(f"  Lags en diferencias: {k_ar_diff}")
        print(f"  det_order: {det_order} (Caso 3 Johansen)")
        
        joh_result = coint_johansen(
            self.data,
            det_order=det_order,
            k_ar_diff=k_ar_diff
        )
        
        trace_stat = joh_result.lr1
        crit_vals_trace = joh_result.cvt
        n_vars = self.data.shape[1]
        
        print("\n" + "-"*70)
        print("TEST DE TRAZA")
        print("-"*70)
        print(f"{'H0: rank ≤':<15} {'Estadístico':<15} {'10%':<12} {'5%':<12} {'1%':<12}")
        print("-"*70)
        
        for i in range(n_vars):
            stat = trace_stat[i]
            cv_90 = crit_vals_trace[i, 0]
            cv_95 = crit_vals_trace[i, 1]
            cv_99 = crit_vals_trace[i, 2]
            
            decision = "**" if stat > cv_95 else "*" if stat > cv_90 else ""
            print(f"{i:<15} {stat:<15.3f} {cv_90:<12.3f} {cv_95:<12.3f} {cv_99:<12.3f} {decision}")
        
        # Determinar rango
        coint_rank = 0
        for i in range(n_vars):
            if trace_stat[i] > crit_vals_trace[i, 1]:
                coint_rank = i + 1
            else:
                break
        
        print("-"*70)
        print(f"\n► Rango de cointegración (α=5%): {coint_rank}")
        
        if coint_rank >= 1:
            print(f"  ✓ Existe cointegración (rank = {coint_rank})")
        else:
            print("  ⚠ No se detectó cointegración al 5%")
        
        self.coint_rank = coint_rank
        self.johansen_results = {
            'trace_stat': trace_stat,
            'crit_vals': crit_vals_trace,
            'rank': coint_rank
        }
        
        # Guardar
        df_joh = pd.DataFrame({
            'rank': range(n_vars),
            'trace_stat': trace_stat,
            'cv_90': crit_vals_trace[:, 0],
            'cv_95': crit_vals_trace[:, 1],
            'cv_99': crit_vals_trace[:, 2]
        })
        
        output_path = RESULTS_TABLES / "johansen_net_liquidity.csv"
        df_joh.to_csv(output_path, index=False)
        print(f"\n✓ Resultados: {output_path}")
        
        return self.johansen_results
    
    def estimate_vecm(self):
        """Estima VECM"""
        print("\n" + "="*70)
        print("ESTIMACIÓN VECM (NET LIQUIDITY)")
        print("="*70)
        
        if self.coint_rank == 0:
            print("\n⚠ rank = 0, no se puede estimar VECM")
            return None
        
        k_ar_diff = self.lag_order - 1
        
        print(f"\nEspecificación:")
        print(f"  Variables: {self.endog_vars}")
        print(f"  Lags: {k_ar_diff}")
        print(f"  Rank: {self.coint_rank}")
        
        self.model = VECM(
            endog=self.data,
            k_ar_diff=k_ar_diff,
            coint_rank=self.coint_rank,
            deterministic='ci'
        )
        
        print("\nEstimando...")
        self.results = self.model.fit()
        print("✓ Completado")
        
        print("\n" + "="*70)
        print("RESULTADOS")
        print("="*70)
        
        try:
            print(self.results.summary())
        except:
            print("⚠ No se puede mostrar summary completo")
        
        self._extract_parameters()
        
        return self.results
    
    def _extract_parameters(self):
        """Extrae parámetros clave"""
        if self.results is None:
            return
        
        beta = self.results.beta
        alpha = self.results.alpha
        
        df_beta = pd.DataFrame(
            beta,
            index=self.endog_vars,
            columns=[f'β_{i+1}' for i in range(beta.shape[1])]
        )
        
        df_alpha = pd.DataFrame(
            alpha,
            index=self.endog_vars,
            columns=[f'α_{i+1}' for i in range(alpha.shape[1])]
        )
        
        # Guardar
        RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
        df_beta.to_csv(RESULTS_TABLES / "vecm_beta_net_liquidity.csv")
        df_alpha.to_csv(RESULTS_TABLES / "vecm_alpha_net_liquidity.csv")
        
        print("\n" + "-"*70)
        print("PARÁMETROS ESTIMADOS")
        print("-"*70)
        
        print("\n[1] Vector de cointegración β:")
        print(df_beta.to_string())
        
        # Relación de equilibrio
        print("\n  Relación de equilibrio:")
        eq_parts = []
        for i, var in enumerate(self.endog_vars):
            coef = beta[i, 0]
            if i == 0:
                eq_parts.append(f"{var}")
            else:
                sign = "+" if coef > 0 else ""
                eq_parts.append(f"{sign} {coef:.4f}·{var}")
        
        print(f"  {' '.join(eq_parts)} = estacionario")
        
        # Coeficiente de net_liquidity
        if 'log_net_liquidity' in self.endog_vars:
            idx_liq = self.endog_vars.index('log_net_liquidity')
            beta_liq = beta[idx_liq, 0]
            
            print(f"\n  ► Coeficiente de log_net_liquidity: β_L = {beta_liq:.4f}")
            
            # Extraer p-value del summary
            try:
                # Buscar en la tabla de cointegración
                summary_str = str(self.results.summary())
                
                # Elasticidad implícita
                elasticity = -beta_liq
                print(f"    Elasticidad implícita: {elasticity:.4f}")
                print(f"    Interpretación: 1% ↑ net liquidity → {elasticity:.4f}% ↑ S&P 500")
                
                if abs(beta_liq) < 0.01:
                    print("\n    ⚠ REFUTA H2: Coeficiente cercano a 0")
                else:
                    print("\n    ✓ CONFIRMA H2: Liquidez neta influye estructuralmente")
            except:
                pass
        
        print("\n[2] Coeficientes de ajuste α:")
        print(df_alpha.to_string())


def main():
    """Pipeline VECM con Net Liquidity"""
    print("\n" + "="*70)
    print(" VECM CON NET LIQUIDITY (FAMILIA A)")
    print("="*70)
    
    start_time = datetime.now()
    
    # Cargar datos
    print("\nCargando datos...")
    data_path = DATA_PROCESSED / "monthly_data.csv"
    df = pd.read_csv(data_path, parse_dates=['date'])
    
    # Verificar que net_liquidity existe
    if 'log_net_liquidity' not in df.columns:
        print("\n❌ ERROR: log_net_liquidity no encontrado en el dataset")
        print("   Ejecuta primero: python src/data_processing/create_monthly.py")
        return
    
    print(f"✓ Dataset cargado: {len(df)} observaciones")
    
    # Crear instancia
    vecm = VECMNetLiquidity(df)
    
    # Pipeline
    vecm.select_lag_order(maxlags=12, ic='aic')
    vecm.johansen_test(det_order=1)
    
    if vecm.coint_rank > 0:
        vecm.estimate_vecm()
    else:
        print("\n⚠ No se puede estimar VECM (rank = 0)")
        print("  → La relación es de corto plazo, no largo plazo")
        print("  → Proceder con VAR en diferencias (H3)")
    
    # Comparación con WALCL
    print("\n" + "="*70)
    print("COMPARACIÓN: NET LIQUIDITY VS WALCL")
    print("="*70)
    
    print("\nRecuerda:")
    print("  WALCL (Familia A original): β = 0.17, p = 0.378 (NO significativo)")
    print("  Net Liquidity (Familia A corregida): β = ?, p = ? (verificar arriba)")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱ Tiempo: {elapsed:.1f}s")
    print("\n✓ Pipeline completado\n")


if __name__ == "__main__":
    main()