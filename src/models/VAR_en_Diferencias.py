"""
Análisis VAR (Vector Autoregression) en diferencias
Implementa IRF con Cholesky y tests de Granger

Autor: Pablo Álvarez de Toledo Rodríguez
Fecha: Marzo 2026
"""

from ast import If
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import select_order
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats
from datetime import datetime
import warnings

# Añadir src/ al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import (
    DATA_PROCESSED,
    RESULTS_TABLES,
    RESULTS_FIGURES,
    FIGURE_DPI
)

warnings.filterwarnings('ignore')


# =============================================================================
# CLASE PRINCIPAL: ANÁLISIS VAR
# =============================================================================

class VARAnalysis:
    """
    Clase para análisis VAR en diferencias
    
    Workflow:
    1. Selección de lags
    2. Estimación del VAR
    3. Tests de Granger causality
    4. Impulse Response Functions (Cholesky)
    5. Forecast Error Variance Decomposition
    6. Diagnósticos
    """
    
    def __init__(self, df, endog_vars=['growth_total_reserves', 'delta_spread', 'delta_vix', 'ret_sp500'], name=''):
        """
        Inicializa análisis VAR
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos mensuales
        endog_vars : list
            Variables endógenas (en diferencias)
            ORDEN IMPORTA para Cholesky:
            - Primero: Más exógena (liquidez)
            - Último: Más endógena (sp500)
        name : str
            Nombre de la especificación
        """
        self.df = df
        self.endog_vars = endog_vars
        self.name = name
        self.data = None
        self.model = None
        self.results = None
        self.lag_order = None
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepara datos para VAR"""
        self.data = self.df[self.endog_vars].dropna()
        
        suffix = f" ({self.name})" if self.name else ""
        
        print(f"\nDatos preparados{suffix}:")
        print(f"  Variables: {self.endog_vars}")
        print(f"  Observaciones: {len(self.data)}")
        
        valid_indices = self.data.index
        dates = self.df.loc[valid_indices, 'date']
        
        print(f"  Periodo: {dates.iloc[0].strftime('%Y-%m')} a "
              f"{dates.iloc[-1].strftime('%Y-%m')}")
        
        print(f"\n  Orden de variables (Cholesky):")
        for i, var in enumerate(self.endog_vars, 1):
            exog_str = "más exógena" if i == 1 else "más endógena" if i == len(self.endog_vars) else "intermedia"
            print(f"    {i}. {var:20s} ({exog_str})")
    
    # =========================================================================
    # PASO 1: SELECCIÓN DE LAGS
    # =========================================================================
    
    def select_lag_order(self, maxlags=12, ic='bic'):
        """Selecciona orden óptimo de lags"""
        print("\n" + "="*70)
        print(f"SELECCIÓN DE ORDEN DE LAGS{' (' + self.name + ')' if self.name else ''}")
        print("="*70)
        
        lag_order_results = select_order(
            self.data,
            maxlags=maxlags,
            deterministic='c'  # Constante
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
        
        if selected < 1:
            print(f"\n⚠ Advertencia: El criterio {ic.upper()} sugiere 0 lags. Se seleccionará 1 lag para garantizar dinámica mínima.")
            selected = 1

        self.lag_order = selected
        print(f"\n► Lags seleccionados ({ic.upper()}): {selected}")
        
        return selected
    
    # =========================================================================
    # PASO 2: ESTIMACIÓN DEL VAR
    # =========================================================================
    
    def estimate_var(self, lags=None):
        """Estima modelo VAR"""
        print("\n" + "="*70)
        print(f"ESTIMACIÓN DEL MODELO VAR{' (' + self.name + ')' if self.name else ''}")
        print("="*70)
        
        if lags is None:
            if self.lag_order is None:
                raise ValueError("Debe ejecutar select_lag_order() primero")
            lags = self.lag_order
        
        if lags < 1:
            print("\n⚠ VAR(0) no permite IRFs ni FEVD. Se ajusta a 1 lag.")
            lags = 1
            
        print(f"\nEspecificación:")
        print(f"  Variables endógenas: {self.endog_vars}")
        print(f"  Lags: {lags}")
        print(f"  Término determinístico: Constante")
        
        # Crear y estimar modelo
        self.model = VAR(self.data)
        self.results = self.model.fit(maxlags=lags, ic=None, trend='c')
        
        print("\n✓ Estimación completada")
        
        # Mostrar resumen
        print("\n" + "="*70)
        print("RESUMEN DE RESULTADOS")
        print("="*70)
        print(self.results.summary())
        
        # Guardar parámetros
        self._save_parameters()
        
        return self.results
    
    def _save_parameters(self):
        """Guarda parámetros estimados"""
        if self.results is None:
            return
        
        # Matriz de coeficientes
        params = self.results.params
        
        suffix = f"_{self.name}" if self.name else ""
        output_path = RESULTS_TABLES / f"var_parameters{suffix}.csv"
        params.to_csv(output_path)
        
        print(f"\n✓ Parámetros guardados: {output_path}")
    
    # =========================================================================
    # PASO 3: GRANGER CAUSALITY
    # =========================================================================
    
    def granger_causality(self, maxlag=None):
        """
        Tests de Granger causality entre variables
        
        Returns
        -------
        dict
            Resultados de tests de causalidad
        """
        print("\n" + "="*70)
        print(f"TESTS DE GRANGER CAUSALITY{' (' + self.name + ')' if self.name else ''}")
        print("="*70)
        
        if maxlag is None:
            maxlag = self.lag_order if self.lag_order else 4
        
        results = {}
        
        # Test para cada par de variables
        print(f"\nTestea: Variable X Granger-causa a Variable Y?")
        print(f"H0: X NO causa (en sentido Granger) a Y")
        print(f"Lags testeados: 1 a {maxlag}\n")
        
        for caused_var in self.endog_vars:
            for causing_var in self.endog_vars:
                if caused_var == causing_var:
                    continue
                
                # Crear DataFrame con solo estas 2 variables
                df_test = self.data[[caused_var, causing_var]].dropna()
                
                # Test de Granger
                try:
                    gc_result = grangercausalitytests(
                        df_test,
                        maxlag=maxlag,
                        verbose=False
                    )
                    
                    # Extraer p-values para cada lag
                    p_values = [gc_result[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1)]
                    min_pval = min(p_values)
                    best_lag = p_values.index(min_pval) + 1
                    
                    # Decisión
                    causes = min_pval < 0.05
                    
                    results[f"{causing_var}_causes_{caused_var}"] = {
                        'min_pval': min_pval,
                        'best_lag': best_lag,
                        'causes': causes
                    }
                    
                    # Imprimir
                    decision = "✓ SÍ causa" if causes else "✗ No causa"
                    print(f"{causing_var:20s} → {caused_var:20s}: p={min_pval:.4f} (lag={best_lag}) {decision}")
                
                except Exception as e:
                    print(f"{causing_var:20s} → {caused_var:20s}: Error - {str(e)}")
        
        # Guardar resultados
        suffix = f"_{self.name}" if self.name else ""
        output_path = RESULTS_TABLES / f"granger_causality{suffix}.csv"
        pd.DataFrame(results).T.to_csv(output_path)
        print(f"\n✓ Resultados guardados: {output_path}")
        
        return results
    
    # =========================================================================
    # PASO 4: IMPULSE RESPONSE FUNCTIONS
    # =========================================================================
    
    def impulse_responses(self, periods=24, orthogonalized=True, plot=True):
        """
        Calcula Impulse Response Functions
        
        Parameters
        ----------
        periods : int
            Horizonte de IRF (meses)
        orthogonalized : bool
            Si True, usa Cholesky (recomendado)
        plot : bool
            Si True, genera gráficos
        
        Returns
        -------
        np.ndarray
            Matriz de IRF
        """
        print("\n" + "="*70)
        print(f"IMPULSE RESPONSE FUNCTIONS{' (' + self.name + ')' if self.name else ''}")
        print("="*70)
        
        if self.results is None:
            raise ValueError("Debe estimar VAR primero")
        
        print(f"\nParámetros:")
        print(f"  Horizonte: {periods} periodos")
        print(f"  Ortogonalizado (Cholesky): {orthogonalized}")
        
        if orthogonalized:
            print(f"\n  Ordenación de Cholesky:")
            print(f"    (Primera variable es más exógena)")
            for i, var in enumerate(self.endog_vars, 1):
                print(f"    {i}. {var}")
        
        # Calcular IRF
        irf = self.results.irf(periods=periods)
        
        # Plotear
        if plot:
            self._plot_irf(irf, periods, orthogonalized)
        
        # Retornar matriz
        if orthogonalized:
            return irf.orth_irfs
        else:
            return irf.irfs
    
    def _plot_irf(self, irf, periods, orthogonalized):
        """Genera gráficos de IRF"""
        try:
            # Crear figura
            fig = irf.plot(
                orth=orthogonalized,
                figsize=(16, 12)
            )
            
            # Título
            orth_str = "Ortogonalizadas (Cholesky)" if orthogonalized else "No ortogonalizadas"
            title_suffix = f" - {self.name}" if self.name else ""
            fig.suptitle(f'Impulse Response Functions - {orth_str}{title_suffix}',
                        fontsize=16, fontweight='bold', y=0.995)
            
            # Guardar
            suffix_file = f"_{self.name}" if self.name else ""
            suffix_orth = "_orth" if orthogonalized else "_nonorth"
            output_path = RESULTS_FIGURES / f"var_irf{suffix_file}{suffix_orth}.png"
            plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
            plt.close()
            
            print(f"\n✓ Gráfico IRF guardado: {output_path}")
            
        except Exception as e:
            print(f"⚠ Error al generar gráfico IRF: {str(e)}")
    

    def plot_irf_single(self, impulse_var, response_var, periods=24, orthogonalized=True):
        """
        Genera gráfico de una IRF específica (para análisis detallado)
        
        Parameters
        ----------
        impulse_var : str
            Variable que recibe el shock
        response_var : str
            Variable cuya respuesta se analiza
        periods : int
            Horizonte
        orthogonalized : bool
            Usar Cholesky
        """
        print(f"\nGenerando IRF: {impulse_var} → {response_var}")
        
        irf = self.results.irf(periods=periods)
        
        # Índices de variables
        impulse_idx = self.endog_vars.index(impulse_var)
        response_idx = self.endog_vars.index(response_var)
        
        # Extraer IRF
        if orthogonalized:
            irf_values = irf.orth_irfs[:, response_idx, impulse_idx]
            try:
                stderr = irf.orth_stderr[:, response_idx, impulse_idx]
            except (AttributeError, TypeError):
                stderr = None
        else:
            irf_values = irf.irfs[:, response_idx, impulse_idx]
            try:
                stderr = irf.stderr[:, response_idx, impulse_idx]
            except (AttributeError, TypeError):
                stderr = None
        
        # Crear gráfico
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(irf_values))
        ax.plot(x, irf_values, linewidth=2, label='IRF')

        if stderr is not None:
            lower = irf_values - 1.96 * stderr
            upper = irf_values + 1.96 * stderr
            ax.fill_between(x, lower, upper, color='blue', alpha=0.3, label='IC 95%')
        else:
            print("⚠ No se pudieron calcular errores estándar para esta IRF")

        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Meses después del shock', fontsize=12)
        ax.set_ylabel('Respuesta', fontsize=12)
        ax.set_title(f'Respuesta de {response_var} a shock en {impulse_var}',
                    fontsize=14, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        
        # Guardar
        suffix = f"_{self.name}" if self.name else ""
        output_path = RESULTS_FIGURES / f"var_irf_{impulse_var}_to_{response_var}{suffix}.png"
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Gráfico guardado: {output_path}")
        
        # Imprimir valores IRF
        print(f"\nValores de IRF:")
        print(f"  h=0:  {irf_values[0]:.6f}")
        print(f"  h=1:  {irf_values[1]:.6f}")
        print(f"  h=3:  {irf_values[3]:.6f}")
        print(f"  h=6:  {irf_values[6]:.6f}")
        print(f"  h=12: {irf_values[12]:.6f}")
        
        
        return irf_values, stderr
    
    # =========================================================================
    # PASO 5: FORECAST ERROR VARIANCE DECOMPOSITION
    # =========================================================================
    
    def variance_decomposition(self, periods=24):
        """
        Descomposición de varianza del error de pronóstico
        
        Returns
        -------
        np.ndarray
            Matriz de descomposición de varianza
        """
        print("\n" + "="*70)
        print(f"FORECAST ERROR VARIANCE DECOMPOSITION{' (' + self.name + ')' if self.name else ''}")
        print("="*70)
        
        if self.results is None:
            raise ValueError("Debe estimar VAR primero")
        
        print(f"\nHorizonte: {periods} periodos")
        
        # Calcular FEVD
        fevd = self.results.fevd(periods=periods)
        
        # Guardar tabla
        self._save_fevd_table(fevd, periods)
        
        # Plotear
        self._plot_fevd(fevd, periods)
        
        return fevd
    
    def _save_fevd_table(self, fevd, periods):
        """Guarda tabla de FEVD"""
        # Crear tabla para cada variable
        for i, var in enumerate(self.endog_vars):
            df_fevd = pd.DataFrame(
                fevd.decomp[:, i, :],
                columns=self.endog_vars
            )
            df_fevd.index.name = 'Horizon'
            
            suffix = f"_{self.name}" if self.name else ""
            output_path = RESULTS_TABLES / f"fevd_{var}{suffix}.csv"
            df_fevd.to_csv(output_path)
        
        print(f"✓ Tablas FEVD guardadas en {RESULTS_TABLES}")
    
    def _plot_fevd(self, fevd, periods):
        """Genera gráficos de FEVD"""
        n_vars = len(self.endog_vars)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for i, var in enumerate(self.endog_vars):
            ax = axes[i]
            
            # Cada fuente de shock en un color diferente
            for j, shock_var in enumerate(self.endog_vars):
                ax.plot(fevd.decomp[:, i, j], label=shock_var, linewidth=2)
            
            ax.set_title(f'Varianza de {var}', fontweight='bold')
            ax.set_xlabel('Horizonte (meses)')
            ax.set_ylabel('Proporción de varianza')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        suffix = f"_{self.name}" if self.name else ""
        output_path = RESULTS_FIGURES / f"var_fevd{suffix}.png"
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Gráfico FEVD guardado: {output_path}")
    
    # =========================================================================
    # PASO 6: DIAGNÓSTICOS
    # =========================================================================
    
    def residual_diagnostics(self):
        """Diagnósticos de residuos"""
        print("\n" + "="*70)
        print(f"DIAGNÓSTICOS DE RESIDUOS{' (' + self.name + ')' if self.name else ''}")
        print("="*70)
        
        if self.results is None:
            raise ValueError("Debe estimar VAR primero")
        
        residuals = self.results.resid
        diagnostics = {}
        
        # Test de Ljung-Box
        print("\nTest de Ljung-Box (Autocorrelación):")
        print(f"{'Variable':<20} {'Q(12)':<12} {'p-value':<12} {'Conclusión'}")
        print("-"*70)
        
        for i, var in enumerate(self.endog_vars):
            resid = residuals.iloc[:, i]
            
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
        
        # Test de Jarque-Bera
        print("\nTest de Jarque-Bera (Normalidad):")
        print(f"{'Variable':<20} {'JB stat':<12} {'p-value':<12} {'Conclusión'}")
        print("-"*70)
        
        for i, var in enumerate(self.endog_vars):
            resid = residuals.iloc[:, i]
            jb_stat, jb_pval = stats.jarque_bera(resid)
            
            conclusion = "✓ Normal" if jb_pval > 0.05 else "✗ No normal"
            print(f"{var:<20} {jb_stat:<12.3f} {jb_pval:<12.4f} {conclusion}")
            
            diagnostics[f'jarquebera_{var}'] = {'stat': jb_stat, 'pval': jb_pval}
        
        # Guardar
        suffix = f"_{self.name}" if self.name else ""
        output_path = RESULTS_TABLES / f"var_diagnostics{suffix}.csv"
        pd.DataFrame(diagnostics).T.to_csv(output_path)
        print(f"\n✓ Diagnósticos guardados: {output_path}")
        
        return diagnostics


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """Pipeline VAR completo"""
    print("\n" + "="*70)
    print(" ANÁLISIS VAR: EFECTOS DE CORTO PLAZO - RESERVAS")
    print("="*70)
    
    start_time = datetime.now()
    
    # Cargar datos
    print("\nCargando datos...")
    data_path = DATA_PROCESSED / "monthly_data.csv"
    df = pd.read_csv(data_path, parse_dates=['date'])

    
    # Verificar que growth_total_reserves existe
    if 'growth_total_reserves' not in df.columns:
        print("\n❌ ERROR: growth_total_reserves no encontrado")
        print("   Ejecuta: python src/data_processing/create_monthly.py")
        return
    
    # Variables para VAR (todas en diferencias/cambios)
    endog_vars = [
        'growth_total_reserves',  # Crecimiento de reservas (más exógeno)
        'delta_spread',     # Cambio en spread BBB
        'delta_vix',        # Cambio en VIX
        'ret_sp500'         # Rendimiento S&P 500 (más endógeno)
    ]
    
    # Crear instancia
    var = VARAnalysis(df, endog_vars=endog_vars, name='reserves')
    
    # Pipeline
    var.select_lag_order(maxlags=12, ic='aic')
    for p in [1, 3, 6]:
        var = VARAnalysis(df, endog_vars=endog_vars, name=f'reserves_var{p}')
        var.estimate_var(lags=p)
        var.granger_causality(maxlag=p)
        var.impulse_responses(periods=24, orthogonalized=True, plot=True)
        var.plot_irf_single('growth_total_reserves', 'ret_sp500', periods=24)
        var.variance_decomposition(periods=24)
        var.residual_diagnostics()

    
    # IRF específica: reserves → sp500
    print("\n" + "="*70)
    print("IRF CLAVE: growth_total_reserves → ret_sp500")
    print("="*70)
    var.plot_irf_single('growth_total_reserves', 'ret_sp500', periods=24)
    
    # FEVD
    var.variance_decomposition(periods=24)
    
    # Diagnósticos
    var.residual_diagnostics()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*70)
    print(f"Tiempo total: {elapsed:.1f}s")
    print("="*70)
    print("\n✓ Pipeline VAR completado\n")


if __name__ == "__main__":
    main()