"""
Estimación del modelo VECM (Vector Error Correction Model)
Implementa test de Johansen y análisis de cointegración

Versión 3: Corregido det_order y muestra completa

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

# Añadir src/ al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import (
    DATA_PROCESSED,
    RESULTS_TABLES,
    RESULTS_FIGURES,
    RESULTS_MODELS,
    VECM_LAG_ORDER,
    VECM_DET_ORDER,
    FIGURE_DPI
)

warnings.filterwarnings('ignore')


# =============================================================================
# CLASE PRINCIPAL: ANÁLISIS VECM
# =============================================================================

class VECMAnalysis:
    """Clase para análisis completo de VECM"""
    
    def __init__(self, df, endog_vars=['log_sp500', 'log_earnings', 'log_balance'], name=''):
        self.df = df
        self.endog_vars = endog_vars
        self.name = name
        self.data = None
        self.model = None
        self.results = None
        self.lag_order = None
        self.coint_rank = None
        self.johansen_results = None
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepara matriz de datos para VECM"""
        self.data = self.df[self.endog_vars].dropna()
        
        suffix = f" ({self.name})" if self.name else ""
        
        print(f"\nDatos preparados{suffix}:")
        print(f"  Variables: {self.endog_vars}")
        print(f"  Observaciones: {len(self.data)}")
        
        valid_indices = self.data.index
        dates = self.df.loc[valid_indices, 'date']
        
        print(f"  Periodo: {dates.iloc[0].strftime('%Y-%m')} a "
              f"{dates.iloc[-1].strftime('%Y-%m')}")
    
    def select_lag_order(self, maxlags=12, ic='bic'):
        """Selecciona orden óptimo de lags"""
        print("\n" + "="*70)
        print(f"SELECCIÓN DE ORDEN DE LAGS{' (' + self.name + ')' if self.name else ''}")
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
        print(f"  FPE:  {lag_order_results.fpe} lags")
        
        if ic == 'aic':
            selected = lag_order_results.aic
        elif ic == 'bic':
            selected = lag_order_results.bic
        elif ic == 'hqic':
            selected = lag_order_results.hqic
        elif ic == 'fpe':
            selected = lag_order_results.fpe
        else:
            raise ValueError(f"Criterio '{ic}' no reconocido")
        
        self.lag_order = selected
        print(f"\n► Lags seleccionados ({ic.upper()}): {selected}")
        print(f"  (En VECM esto corresponde a k_ar_diff = {selected - 1})")
        
        return selected
    
    def johansen_test(self, det_order=1, k_ar_diff=None):  # ← CORRECCIÓN: det_order=1
        """Realiza test de Johansen para cointegración"""
        print("\n" + "="*70)
        print(f"TEST DE JOHANSEN PARA COINTEGRACIÓN{' (' + self.name + ')' if self.name else ''}")
        print("="*70)
        
        if k_ar_diff is None:
            if self.lag_order is None:
                raise ValueError("Debe ejecutar select_lag_order() primero")
            k_ar_diff = self.lag_order - 1
        
        print(f"\nParámetros del test:")
        print(f"  Lags en diferencias (k_ar_diff): {k_ar_diff}")
        print(f"  Orden determinístico (det_order): {det_order}")
        print(f"    (Caso 3 Johansen: constante no restringida)")
        
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
            
            if stat > cv_95:
                decision = "**"
            elif stat > cv_90:
                decision = "*"
            else:
                decision = ""
            
            print(f"{i:<15} {stat:<15.3f} {cv_90:<12.3f} {cv_95:<12.3f} {cv_99:<12.3f} {decision}")
        
        coint_rank_5 = 0
        coint_rank_10 = 0
        
        for i in range(n_vars):
            if trace_stat[i] > crit_vals_trace[i, 1]:
                coint_rank_5 = i + 1
            else:
                break
        
        for i in range(n_vars):
            if trace_stat[i] > crit_vals_trace[i, 0]:
                coint_rank_10 = i + 1
            else:
                break
        
        print("-"*70)
        print(f"\n► Rango de cointegración (α=5%):  {coint_rank_5}")
        print(f"► Rango de cointegración (α=10%): {coint_rank_10}")
        
        if coint_rank_5 == 0 and coint_rank_10 == 0:
            print("\n  ⚠ No se detectó cointegración")
            print("  Recomendación: Usar VAR en diferencias")
            self.coint_rank = 0
        elif coint_rank_5 == 0 and coint_rank_10 > 0:
            print(f"\n  ⚠ Cointegración borderline:")
            print(f"    - No significativa al 5%")
            print(f"    - Significativa al 10% (r={coint_rank_10})")
            print(f"  Posibles causas: Cambios estructurales, muestra pequeña")
            print(f"  Recomendación: Proceder con r={coint_rank_10} con cautela")
            self.coint_rank = coint_rank_10
        elif coint_rank_5 == 1:
            print(f"  ✓ Existe 1 relación de cointegración")
            print(f"  Recomendación: Estimar VECM con r=1")
            self.coint_rank = 1
        else:
            print(f"  ✓ Existen {coint_rank_5} relaciones de cointegración")
            print(f"  Recomendación: Estimar VECM con r={coint_rank_5}")
            self.coint_rank = coint_rank_5
        
        self.johansen_results = {
            'n_vars': n_vars,
            'k_ar_diff': k_ar_diff,
            'det_order': det_order,
            'trace_stat': trace_stat,
            'crit_vals_trace': crit_vals_trace,
            'coint_rank_5': coint_rank_5,
            'coint_rank_10': coint_rank_10
        }
        
        df_johansen = pd.DataFrame({
            'rank': range(n_vars),
            'trace_stat': trace_stat,
            'cv_90': crit_vals_trace[:, 0],
            'cv_95': crit_vals_trace[:, 1],
            'cv_99': crit_vals_trace[:, 2]
        })
        
        suffix = f"_{self.name}" if self.name else ""
        output_path = RESULTS_TABLES / f"johansen_test{suffix}.csv"
        df_johansen.to_csv(output_path, index=False)
        print(f"\n✓ Resultados guardados: {output_path}")
        
        return self.johansen_results
    
    def estimate_vecm(self, k_ar_diff=None, coint_rank=None, deterministic='ci'):
        """Estima modelo VECM"""
        print("\n" + "="*70)
        print(f"ESTIMACIÓN DEL MODELO VECM{' (' + self.name + ')' if self.name else ''}")
        print("="*70)
        
        if k_ar_diff is None:
            if self.lag_order is None:
                raise ValueError("Debe ejecutar select_lag_order() primero")
            k_ar_diff = self.lag_order - 1
        
        if coint_rank is None:
            if self.coint_rank is None:
                raise ValueError("Debe ejecutar johansen_test() primero")
            coint_rank = self.coint_rank
        
        if coint_rank == 0:
            print("\n⚠ ADVERTENCIA: Rango de cointegración = 0")
            print("  No se puede estimar VECM sin cointegración")
            print("  Alternativa: Usar VAR en diferencias")
            return None
        
        print(f"\nEspecificación:")
        print(f"  Variables endógenas: {self.endog_vars}")
        print(f"  Lags en diferencias: {k_ar_diff}")
        print(f"  Rango de cointegración: {coint_rank}")
        print(f"  Término determinístico: {deterministic}")
        
        self.model = VECM(
            endog=self.data,
            k_ar_diff=k_ar_diff,
            coint_rank=coint_rank,
            deterministic=deterministic
        )
        
        print("\nEstimando parámetros...")
        self.results = self.model.fit()
        
        print("✓ Estimación completada")
        
        print("\n" + "="*70)
        print("RESUMEN DE RESULTADOS")
        print("="*70)
        
        try:
            print(self.results.summary())
        except:
            print("⚠ No se puede mostrar summary completo")
        
        self._extract_parameters()
        
        return self.results
    
    def _extract_parameters(self):
        """Extrae y guarda parámetros clave del VECM"""
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
        
        suffix = f"_{self.name}" if self.name else ""
        output_beta = RESULTS_TABLES / f"vecm_beta{suffix}.csv"
        output_alpha = RESULTS_TABLES / f"vecm_alpha{suffix}.csv"
        
        df_beta.to_csv(output_beta)
        df_alpha.to_csv(output_alpha)
        
        print(f"\n✓ Vectores de cointegración: {output_beta}")
        print(f"✓ Coeficientes de ajuste: {output_alpha}")
        
        print("\n" + "-"*70)
        print("PARÁMETROS ESTIMADOS")
        print("-"*70)
        
        print("\n[1] Vector de cointegración β (normalizado en primera variable):")
        print(df_beta.to_string())
        
        print("\n  Relación de equilibrio de largo plazo:")
        eq_parts = []
        for i, var in enumerate(self.endog_vars):
            coef = beta[i, 0]
            if i == 0:
                eq_parts.append(f"{var}")
            else:
                sign = "+" if coef > 0 else ""
                eq_parts.append(f"{sign} {coef:.4f}·{var}")
        
        print(f"  {' '.join(eq_parts)} = estacionario")
        
        if 'log_balance' in self.endog_vars:
            idx_balance = self.endog_vars.index('log_balance')
            beta_balance = beta[idx_balance, 0]
            
            print(f"\n  ► Coeficiente de log_balance: β_b = {beta_balance:.4f}")
            
            if abs(beta_balance) < 0.01:
                print("    Interpretación: Balance NO tiene efecto estructural significativo")
                print("    ⚠ REFUTA Hipótesis 2")
            else:
                elasticity = -beta_balance
                print(f"    Elasticidad implícita: {elasticity:.4f}")
                print(f"    Interpretación: 1% ↑ balance → {elasticity:.4f}% ↑ S&P 500 (LP)")
                if abs(elasticity - 1.0) < 0.2:
                    print("    ✓ Elasticidad cercana a 1 (como literatura sugiere)")
                print("    ✓ CONFIRMA Hipótesis 2 (liquidez influye estructuralmente)")
        
        print("\n[2] Coeficientes de ajuste α:")
        print(df_alpha.to_string())
        
        print("\n  Interpretación:")
        for i, var in enumerate(self.endog_vars):
            alpha_val = alpha[i, 0]
            if abs(alpha_val) < 0.01:
                interp = f"No se ajusta (weakly exogenous)"
            elif alpha_val < 0:
                half_life = np.log(0.5) / np.log(1 + alpha_val)
                interp = f"Corrige desviaciones (α={alpha_val:.4f}, vida media={half_life:.1f} meses)"
            else:
                interp = f"Amplifica desviaciones (α={alpha_val:.4f}, ¡inestable!)"
            print(f"    {var:15s}: {interp}")
    
    def residual_diagnostics(self):
        """Diagnósticos de residuos"""
        print("\n" + "="*70)
        print(f"DIAGNÓSTICOS DE RESIDUOS{' (' + self.name + ')' if self.name else ''}")
        print("="*70)
        
        if self.results is None:
            print("⚠ No hay modelo estimado")
            return None
        
        residuals = self.results.resid
        diagnostics = {}
        
        print("\nTest de Ljung-Box (Autocorrelación):")
        print(f"{'Variable':<20} {'Q(12)':<12} {'p-value':<12} {'Conclusión'}")
        print("-"*70)
        
        for i, var in enumerate(self.endog_vars):
            resid = residuals[:, i]
            
            try:
                lb_result = acorr_ljungbox(resid, lags=[12], return_df=True)
                lb_stat = lb_result['lb_stat'].iloc[0]
                lb_pval = lb_result['lb_pvalue'].iloc[0]
            except:
                lb_result = acorr_ljungbox(resid, lags=[12], return_df=False)
                lb_stat = lb_result[0][0] if hasattr(lb_result[0], '__getitem__') else lb_result[0]
                lb_pval = lb_result[1][0] if hasattr(lb_result[1], '__getitem__') else lb_result[1]
            
            conclusion = "✓ No autocorr" if lb_pval > 0.05 else "✗ Autocorr"
            print(f"{var:<20} {lb_stat:<12.3f} {lb_pval:<12.4f} {conclusion}")
            
            diagnostics[f'ljungbox_{var}'] = {'stat': lb_stat, 'pval': lb_pval}
        
        print("\nTest de Jarque-Bera (Normalidad):")
        print(f"{'Variable':<20} {'JB stat':<12} {'p-value':<12} {'Conclusión'}")
        print("-"*70)
        
        for i, var in enumerate(self.endog_vars):
            resid = residuals[:, i]
            jb_stat, jb_pval = stats.jarque_bera(resid)
            
            conclusion = "✓ Normal" if jb_pval > 0.05 else "✗ No normal"
            print(f"{var:<20} {jb_stat:<12.3f} {jb_pval:<12.4f} {conclusion}")
            
            diagnostics[f'jarquebera_{var}'] = {'stat': jb_stat, 'pval': jb_pval}
        
        suffix = f"_{self.name}" if self.name else ""
        output_path = RESULTS_TABLES / f"vecm_diagnostics{suffix}.csv"
        pd.DataFrame(diagnostics).T.to_csv(output_path)
        print(f"\n✓ Diagnósticos guardados: {output_path}")
        
        return diagnostics
    
    def impulse_responses(self, periods=24, orthogonalized=True, plot=True):
        """Calcula Impulse Response Functions"""
        print("\n" + "="*70)
        print(f"IMPULSE RESPONSE FUNCTIONS{' (' + self.name + ')' if self.name else ''}")
        print("="*70)
        
        if self.results is None:
            print("⚠ No hay modelo estimado")
            return None
        
        print(f"\nParámetros:")
        print(f"  Horizonte: {periods} periodos")
        print(f"  Ortogonalizado (Cholesky): {orthogonalized}")
        
        irf = self.results.irf(periods=periods)
        
        if orthogonalized:
            irf_matrix = irf.orth_irfs
        else:
            irf_matrix = irf.irfs
        
        if plot:
            self._plot_irfs(irf, periods, orthogonalized)
        
        return irf_matrix
    
    def _plot_irfs(self, irf, periods, orthogonalized):
        """Genera gráficos de IRF"""
        try:
            fig = irf.plot(
                impulse=self.endog_vars,
                response=self.endog_vars,
                orth=orthogonalized,
                figsize=(14, 10)
            )
            
            orth_str = "Ortogonalizadas (Cholesky)" if orthogonalized else "No ortogonalizadas"
            title_suffix = f" - {self.name}" if self.name else ""
            fig.suptitle(f'Impulse Response Functions - {orth_str}{title_suffix}', 
                         fontsize=16, fontweight='bold', y=0.995)
            
            suffix_file = f"_{self.name}" if self.name else ""
            suffix_orth = "_orth" if orthogonalized else "_nonorth"
            output_path = RESULTS_FIGURES / f"vecm_irf{suffix_file}{suffix_orth}.png"
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Gráfico IRF guardado: {output_path}")
        except Exception as e:
            print(f"⚠ No se pudo generar gráfico IRF: {str(e)}")


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Pipeline con especificación única (muestra completa)"""
    print("\n" + "="*70)
    print(" ANÁLISIS VECM: MUESTRA COMPLETA (2000-2026)")
    print("="*70)
    
    start_time = datetime.now()
    
    print("\nCargando datos...")
    data_path = DATA_PROCESSED / "monthly_data.csv"
    df = pd.read_csv(data_path, parse_dates=['date'])
    
    # SIN FILTRO DE FECHA - Usar muestra completa
    print(f"✓ Muestra completa cargada: {len(df)} observaciones")
    
    # Variables del VECM
    endog_vars = ['log_sp500', 'log_earnings', 'log_balance']
    
    # Crear instancia
    vecm = VECMAnalysis(df, endog_vars=endog_vars, name='full_sample')
    
    # Pipeline
    vecm.select_lag_order(maxlags=12, ic='bic')
    vecm.johansen_test(det_order=1)  # ← CORRECCIÓN: det_order=1
    
    if vecm.coint_rank > 0:
        vecm.estimate_vecm()
        vecm.residual_diagnostics()
        vecm.impulse_responses(periods=24, plot=True)
    else:
        print("\n⚠ No se detectó cointegración")
        print("  Recomendación: Proceder con VAR en diferencias (H3)")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*70)
    print(f"Tiempo total: {elapsed:.1f}s")
    print("="*70)
    print("\n✓ Pipeline completado\n")


if __name__ == "__main__":
    main()