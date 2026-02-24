"""
Estimación del modelo VECM (Vector Error Correction Model)
Implementa test de Johansen y análisis de cointegración

Autor: Pablo Álvarez de Toledo Rodríguez
Fecha: Enero 2026
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank, coint_johansen
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from datetime import datetime

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


# CLASE PRINCIPAL: ANÁLISIS VECM
class VECMAnalysis:
    """
    Clase para análisis completo de VECM
    
    Flujo de trabajo:
    1. Selección de orden de lags
    2. Test de Johansen para cointegración
    3. Estimación del VECM
    4. Diagnósticos de residuos
    5. Impulse Response Functions
    6. Análisis de descomposición de varianza
    """
    
    def __init__(self, df, endog_vars=['log_sp500', 'log_earnings', 'log_balance']):
        """
        Inicializa análisis VECM
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame con datos mensuales
        endog_vars : list
            Lista de variables endógenas (en orden)
            
        Notes
        -----
        El orden de las variables importa:
        - Primera variable: Será normalizada a 1 en vector de cointegración
        - En nuestro caso: log_sp500 (variable dependiente principal)
        """
        self.df = df
        self.endog_vars = endog_vars
        self.data = None
        self.model = None
        self.results = None
        self.lag_order = None
        self.coint_rank = None
        
        # Preparar datos
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepara matriz de datos para VECM"""
        # Seleccionar variables y eliminar NaN
        self.data = self.df[self.endog_vars].dropna()
        
        print(f"\nDatos preparados:")
        print(f"  Variables: {self.endog_vars}")
        print(f"  Observaciones: {len(self.data)}")
        print(f"  Periodo: {self.df.loc[self.data.index[0], 'date'].strftime('%Y-%m')} a "
              f"{self.df.loc[self.data.index[-1], 'date'].strftime('%Y-%m')}")
    

    # PASO 1: SELECCIÓN DE LAGS
    
    def select_lag_order(self, maxlags=12, ic='aic'):
        """
        Selecciona orden óptimo de lags usando criterios de información
        
        Parameters
        ----------
        maxlags : int
            Número máximo de lags a considerar
        ic : str
            Criterio de información: 'aic', 'bic', 'hqic', 'fpe'
        
        Returns
        -------
        int
            Orden de lags seleccionado
            
        Notes
        -----
        Criterios de información:
        - AIC (Akaike): Penaliza poco la complejidad, tiende a sobreparametrizar
        - BIC (Schwarz): Penaliza más, prefiere modelos parsimoniosos
        - HQIC (Hannan-Quinn): Intermedio entre AIC y BIC
        
        Para series financieras mensuales:
        - 1-3 lags suele ser suficiente
        - Más de 6 lags raramente mejora
        
        En VECM, si seleccionas p lags en VAR → VECM tendrá p-1 lags en diferencias
        """
        print("\n" + "="*70)
        print("SELECCIÓN DE ORDEN DE LAGS")
        print("="*70)
        
        # Estimar VAR sin restricciones para selección de lags
        lag_order_results = select_order(
            self.data,
            maxlags=maxlags,
            deterministic='ci'  # Constante dentro de cointegración
        )
        
        # Extraer órdenes óptimos según cada criterio
        print("\nCriterios de información:")
        print(f"  AIC:  {lag_order_results.aic} lags")
        print(f"  BIC:  {lag_order_results.bic} lags")
        print(f"  HQIC: {lag_order_results.hqic} lags")
        print(f"  FPE:  {lag_order_results.fpe} lags")
        
        # Seleccionar según criterio especificado
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
    

    # PASO 2: TEST DE JOHANSEN
    
    def johansen_test(self, det_order=0, k_ar_diff=None):
        """
        Realiza test de Johansen para cointegración
        
        Parameters
        ----------
        det_order : int
            Orden del término determinístico:
            -1: Sin constante ni tendencia
             0: Constante en relación de cointegración (DEFAULT)
             1: Constante fuera + dentro de cointegración
        k_ar_diff : int
            Lags en diferencias. Si None, usa self.lag_order - 1
        
        Returns
        -------
        dict
            Resultados del test de Johansen
            
        Notes
        -----
        Casos de Johansen:
        - Caso 1 (det_order=-1): Sin tendencia, sin constante
          Raro en finanzas
          
        - Caso 2 (det_order=0): Constante restringida
          Relación: X_t = α + β'Y_t + estacionario
          SIN tendencia en niveles
          RECOMENDADO para precios de activos
          
        - Caso 3 (det_order=1): Constante no restringida
          Relación: X_t = β'Y_t + estacionario
          CON tendencia lineal en niveles
          Usar si variables tienen clara tendencia
        
        Test de traza:
        H0: rank ≤ r  vs  H1: rank > r
        
        Si test rechaza con r=0 pero no rechaza con r=1:
        → Existe 1 relación de cointegración
        """
        print("\n" + "="*70)
        print("TEST DE JOHANSEN PARA COINTEGRACIÓN")
        print("="*70)
        
        if k_ar_diff is None:
            if self.lag_order is None:
                raise ValueError("Debe ejecutar select_lag_order() primero")
            k_ar_diff = self.lag_order - 1
        
        print(f"\nParámetros del test:")
        print(f"  Lags en diferencias (k_ar_diff): {k_ar_diff}")
        print(f"  Orden determinístico (det_order): {det_order}")
        
        # Realizar test de Johansen
        joh_result = coint_johansen(
            self.data,
            det_order=det_order,
            k_ar_diff=k_ar_diff
        )
        
        # Extraer estadísticos
        trace_stat = joh_result.lr1  # Estadístico de traza
        max_eig_stat = joh_result.lr2  # Estadístico de máximo eigenvalor
        crit_vals_trace = joh_result.cvt  # Valores críticos traza (90%, 95%, 99%)
        crit_vals_eig = joh_result.cvm  # Valores críticos eigenvalor
        
        # Número de variables
        n_vars = self.data.shape[1]
        
        # Tabla de resultados - Test de traza
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
            
            # Determinar si rechaza
            if stat > cv_95:
                decision = "**"  # Rechaza al 5%
            elif stat > cv_90:
                decision = "*"   # Rechaza al 10%
            else:
                decision = ""
            
            print(f"{i:<15} {stat:<15.3f} {cv_90:<12.3f} {cv_95:<12.3f} {cv_99:<12.3f} {decision}")
        
        # Determinar rango de cointegración
        coint_rank = 0
        for i in range(n_vars):
            if trace_stat[i] > crit_vals_trace[i, 1]:  # Comparar con 95%
                coint_rank = i + 1
            else:
                break
        
        print("-"*70)
        print(f"\n► Rango de cointegración detectado (α=5%): {coint_rank}")
        
        if coint_rank == 0:
            print("  ⚠ No se detectó cointegración")
            print("  Recomendación: Usar VAR en diferencias")
        elif coint_rank == 1:
            print("  ✓ Existe 1 relación de cointegración")
            print("  Recomendación: Estimar VECM con r=1")
        else:
            print(f"  ✓ Existen {coint_rank} relaciones de cointegración")
            print(f"  Recomendación: Estimar VECM con r={coint_rank}")
        
        self.coint_rank = coint_rank
        
        # Guardar resultados en diccionario
        results_dict = {
            'n_vars': n_vars,
            'k_ar_diff': k_ar_diff,
            'det_order': det_order,
            'trace_stat': trace_stat,
            'crit_vals_trace_90': crit_vals_trace[:, 0],
            'crit_vals_trace_95': crit_vals_trace[:, 1],
            'crit_vals_trace_99': crit_vals_trace[:, 2],
            'coint_rank': coint_rank
        }
        
        # Guardar como CSV
        df_johansen = pd.DataFrame({
            'rank': range(n_vars),
            'trace_stat': trace_stat,
            'cv_90': crit_vals_trace[:, 0],
            'cv_95': crit_vals_trace[:, 1],
            'cv_99': crit_vals_trace[:, 2]
        })
        
        output_path = RESULTS_TABLES / "johansen_test.csv"
        df_johansen.to_csv(output_path, index=False)
        print(f"\n✓ Resultados guardados: {output_path}")
        
        return results_dict
    

    # PASO 3: ESTIMACIÓN DEL VECM
    
    def estimate_vecm(self, k_ar_diff=None, coint_rank=None, deterministic='ci'):
        """
        Estima modelo VECM
        
        Parameters
        ----------
        k_ar_diff : int
            Lags en diferencias
        coint_rank : int
            Rango de cointegración
        deterministic : str
            Término determinístico:
            'nc': Sin constante
            'co': Constante fuera de cointegración
            'ci': Constante dentro de cointegración (DEFAULT)
            'lo': Tendencia lineal fuera
            'li': Tendencia lineal dentro
        
        Returns
        -------
        VECMResults
            Objeto con resultados del VECM
            
        Notes
        -----
        El VECM estima:
        ΔX_t = αβ'X_{t-1} + Γ₁ΔX_{t-1} + ... + ε_t
        
        Parámetros estimados:
        - α: Matriz de ajuste (n_vars × coint_rank)
        - β: Vectores de cointegración (n_vars × coint_rank)
        - Γ: Matrices de corto plazo
        """
        print("\n" + "="*70)
        print("ESTIMACIÓN DEL MODELO VECM")
        print("="*70)
        
        if k_ar_diff is None:
            if self.lag_order is None:
                raise ValueError("Debe ejecutar select_lag_order() primero")
            k_ar_diff = self.lag_order - 1
        
        if coint_rank is None:
            if self.coint_rank is None:
                raise ValueError("Debe ejecutar johansen_test() primero")
            coint_rank = self.coint_rank
        
        print(f"\nEspecificación:")
        print(f"  Variables endógenas: {self.endog_vars}")
        print(f"  Lags en diferencias: {k_ar_diff}")
        print(f"  Rango de cointegración: {coint_rank}")
        print(f"  Término determinístico: {deterministic}")
        
        # Crear modelo VECM
        self.model = VECM(
            endog=self.data,
            k_ar_diff=k_ar_diff,
            coint_rank=coint_rank,
            deterministic=deterministic
        )
        
        # Estimar
        print("\nEstimando parámetros...")
        self.results = self.model.fit()
        
        print("✓ Estimación completada")
        
        # Mostrar resumen
        print("\n" + "="*70)
        print("RESUMEN DE RESULTADOS")
        print("="*70)
        if coint_rank > 0:
            print(self.results.summary())
        else:
            print("\n⚠ No se puede mostrar summary() con rango de cointegración = 0")
            print("  El modelo estimado es equivalente a un VAR en diferencias")
        
        # Extraer y guardar parámetros clave
        self._extract_parameters()
        
        return self.results
    
    def _extract_parameters(self):
        """Extrae y guarda parámetros clave del VECM"""
        # Vector de cointegración (normalizado)
        beta = self.results.beta
        
        # Matriz de ajuste
        alpha = self.results.alpha
        
        # Crear DataFrame con vectores de cointegración
        df_beta = pd.DataFrame(
            beta,
            index=self.endog_vars,
            columns=[f'β_{i+1}' for i in range(beta.shape[1])]
        )
        
        # Crear DataFrame con coeficientes de ajuste
        df_alpha = pd.DataFrame(
            alpha,
            index=self.endog_vars,
            columns=[f'α_{i+1}' for i in range(alpha.shape[1])]
        )
        
        # Guardar
        output_beta = RESULTS_TABLES / "vecm_cointegration_vectors.csv"
        output_alpha = RESULTS_TABLES / "vecm_adjustment_coefficients.csv"
        
        df_beta.to_csv(output_beta)
        df_alpha.to_csv(output_alpha)
        
        print(f"\n✓ Vectores de cointegración guardados: {output_beta}")
        print(f"✓ Coeficientes de ajuste guardados: {output_alpha}")
        
        # Imprimir interpretación
        print("\n" + "-"*70)
        print("INTERPRETACIÓN DE PARÁMETROS")
        print("-"*70)
        
        print("\nVector de cointegración (β, normalizado):")
        print(df_beta)
        
        print("\n  Relación de equilibrio de largo plazo:")
        eq_parts = []
        for i, var in enumerate(self.endog_vars):
            coef = beta[i, 0]
            if i == 0:
                eq_parts.append(f"{var}")
            else:
                sign = "+" if coef > 0 else "-"
                eq_parts.append(f"{sign} {abs(coef):.3f}·{var}")
        
        print(f"  {' '.join(eq_parts)} = estacionario")
        
        print("\nCoeficientes de ajuste (α):")
        print(df_alpha)
        
        print("\n  Interpretación:")
        for i, var in enumerate(self.endog_vars):
            alpha_val = alpha[i, 0]
            if abs(alpha_val) < 0.01:
                interp = f"{var} no se ajusta (weakly exogenous)"
            elif alpha_val < 0:
                interp = f"{var} corrige desviaciones (α={alpha_val:.3f})"
            else:
                interp = f"{var} amplifica desviaciones (α={alpha_val:.3f}, inestable)"
            print(f"    {interp}")
    

    # PASO 4: DIAGNÓSTICOS
    
    def residual_diagnostics(self):
        """
        Realiza diagnósticos de los residuos del VECM
        
        Tests realizados:
        1. Ljung-Box: Autocorrelación
        2. Jarque-Bera: Normalidad
        3. Heterocedasticidad (visual)
        
        Returns
        -------
        dict
            Resultados de los tests
        """
        print("\n" + "="*70)
        print("DIAGNÓSTICOS DE RESIDUOS")
        print("="*70)
        
        if self.results is None:
            raise ValueError("Debe estimar VECM primero")
        
        # Obtener residuos
        residuals = self.results.resid
        
        diagnostics = {}
        
        # Test de Ljung-Box para cada ecuación
        print("\nTest de Ljung-Box (Autocorrelación):")
        print(f"{'Variable':<20} {'Q(12)':<12} {'p-value':<12} {'Conclusión'}")
        print("-"*70)
        
        for i, var in enumerate(self.endog_vars):
            resid = residuals[:, i]
            lb_result = acorr_ljungbox(resid, lags=[12], return_df=False)
            lb_stat = lb_result[0][0]
            lb_pval = lb_result[1][0]
            
            conclusion = "✓ No autocorr" if lb_pval > 0.05 else "✗ Autocorr detectada"
            
            print(f"{var:<20} {lb_stat:<12.3f} {lb_pval:<12.4f} {conclusion}")
            
            diagnostics[f'ljungbox_{var}'] = {'stat': lb_stat, 'pval': lb_pval}
        
        # Test de Jarque-Bera (normalidad)
        print("\nTest de Jarque-Bera (Normalidad):")
        print(f"{'Variable':<20} {'JB stat':<12} {'p-value':<12} {'Conclusión'}")
        print("-"*70)
        
        for i, var in enumerate(self.endog_vars):
            resid = residuals[:, i]
            jb_stat, jb_pval = stats.jarque_bera(resid)
            
            conclusion = "✓ Normal" if jb_pval > 0.05 else "✗ No normal"
            
            print(f"{var:<20} {jb_stat:<12.3f} {jb_pval:<12.4f} {conclusion}")
            
            diagnostics[f'jarquebera_{var}'] = {'stat': jb_stat, 'pval': jb_pval}
        
        # Guardar diagnósticos
        output_path = RESULTS_TABLES / "vecm_diagnostics.csv"
        pd.DataFrame(diagnostics).T.to_csv(output_path)
        print(f"\n✓ Diagnósticos guardados: {output_path}")
        
        return diagnostics
    
    # PASO 5: IMPULSE RESPONSE FUNCTIONS
    
    def impulse_responses(self, periods=24, impulse=None, response=None, 
                         orthogonalized=True, plot=True):
        """
        Calcula Impulse Response Functions
        
        Parameters
        ----------
        periods : int
            Horizonte de la IRF (meses)
        impulse : str or None
            Variable que recibe el shock. Si None, todas.
        response : str or None
            Variable cuya respuesta se analiza. Si None, todas.
        orthogonalized : bool
            Si True, usa Cholesky para ortogonalizar shocks
        plot : bool
            Si True, genera gráficos
        
        Returns
        -------
        np.ndarray
            Matriz de IRF (periods × n_vars × n_vars)
            
        Notes
        -----
        IRF mide: ¿Cómo responde Y a un shock en X?
        
        Ortogonalización (Cholesky):
        - Orden importa: Primera variable es "más exógena"
        - En nuestro caso: [log_sp500, log_earnings, log_balance]
        - Asume: Balance puede afectar S&P contemporáneamente
                 pero S&P NO afecta Balance contemporáneamente
        
        Para tu H3, la IRF relevante es:
        Impulse: log_balance
        Response: log_sp500
        
        Esperamos: IRF positiva (shock liquidez → sube S&P)
        """
        print("\n" + "="*70)
        print("IMPULSE RESPONSE FUNCTIONS")
        print("="*70)
        
        if self.results is None:
            raise ValueError("Debe estimar VECM primero")
        
        print(f"\nParámetros:")
        print(f"  Horizonte: {periods} periodos")
        print(f"  Ortogonalizado (Cholesky): {orthogonalized}")
        
        # Calcular IRF
        irf = self.results.irf(periods=periods)
        
        if orthogonalized:
            irf_matrix = irf.orth_irfs
        else:
            irf_matrix = irf.irfs
        
        # Plotear si se solicita
        if plot:
            self._plot_irfs(irf, periods, orthogonalized)
        
        return irf_matrix
    
    def _plot_irfs(self, irf, periods, orthogonalized):
        """Genera gráficos de IRF"""
        # Crear figura
        fig = irf.plot(
            impulse=self.endog_vars,
            response=self.endog_vars,
            orth=orthogonalized,
            figsize=(14, 10)
        )
        
        # Título
        orth_str = "Ortogonalizadas (Cholesky)" if orthogonalized else "No ortogonalizadas"
        fig.suptitle(f'Impulse Response Functions - {orth_str}', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        # Guardar
        suffix = "_orth" if orthogonalized else "_nonorth"
        output_path = RESULTS_FIGURES / f"vecm_irf{suffix}.png"
        plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Gráfico IRF guardado: {output_path}")


# FUNCIÓN PRINCIPAL

def main():
    """Pipeline completo de análisis VECM"""
    print("\n" + "="*70)
    print(" ANÁLISIS VECM: COINTEGRACIÓN ENTRE LIQUIDEZ Y VALORACIÓN")
    print("="*70)
    
    start_time = datetime.now()
    
    # Cargar datos
    print("\nCargando datos...")
    data_path = DATA_PROCESSED / "monthly_data.csv"
    df = pd.read_csv(data_path, parse_dates=['date'])
    
    # Variables del VECM
    endog_vars = ['log_sp500', 'log_earnings', 'log_balance']
    
    # Crear instancia de análisis
    vecm = VECMAnalysis(df, endog_vars=endog_vars)
    
    # PASO 1: Seleccionar lags
    vecm.select_lag_order(maxlags=12, ic='bic')
    
    # PASO 2: Test de Johansen
    vecm.johansen_test(det_order=0)
    
    # PASO 3: Estimar VECM
    vecm.estimate_vecm()
    
    # PASO 4: Diagnósticos
    vecm.residual_diagnostics()
    
    # PASO 5: IRF
    vecm.impulse_responses(periods=24, orthogonalized=True, plot=True)
    
    # Resumen final
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*70)
    print("ANÁLISIS VECM COMPLETADO")
    print("="*70)
    print(f"Tiempo total: {elapsed:.1f}s")
    print(f"\nResultados guardados en:")
    print(f"  Tablas: {RESULTS_TABLES}")
    print(f"  Figuras: {RESULTS_FIGURES}")
    print("\n✓ Pipeline completado\n")


# EJECUCIÓN

if __name__ == "__main__":
    main()