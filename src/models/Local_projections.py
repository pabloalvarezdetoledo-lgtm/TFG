"""
Local Projections (Jordà, 2005)
Estimación de respuestas dinámicas sin imponer la estructura completa de un VAR.

Autor: Pablo Álvarez de Toledo Rodríguez
Fecha: Marzo 2026
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Añadir src/ al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import (
    DATA_PROCESSED,
    RESULTS_TABLES,
    RESULTS_FIGURES,
    FIGURE_DPI
)

warnings.filterwarnings("ignore")


class LocalProjectionAnalysis:
    """
    Clase para estimar Local Projections.

    Ecuación base:
        y_{t+h} = alpha_h + beta_h * shock_t + Gamma_h' X_t + epsilon_{t+h}

    donde:
        - y_{t+h}: variable respuesta en horizonte h
        - shock_t: proxy de shock de liquidez
        - X_t: controles contemporáneos / rezagados
        - beta_h: IRF estimada directamente
    """

    def __init__(
        self,
        df: pd.DataFrame,
        shock_var: str,
        response_vars: list[str],
        control_vars: list[str],
        lags: int = 6,
        horizons: int = 24,
        name: str = ""
    ):
        self.df = df.copy()
        self.shock_var = shock_var
        self.response_vars = response_vars
        self.control_vars = control_vars
        self.lags = lags
        self.horizons = horizons
        self.name = name

        self.results = {}

        self._validate_inputs()
        self._prepare_base_data()

    def _validate_inputs(self):
        needed = [self.shock_var] + self.response_vars + self.control_vars + ["date"]
        missing = [c for c in needed if c not in self.df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en el DataFrame: {missing}")

    def _prepare_base_data(self):
        """
    Mantiene solo columnas relevantes, sin duplicados y ordenadas por fecha.
    """
        keep_cols = ["date"]

        for col in [self.shock_var] + self.response_vars + self.control_vars:
            if col not in keep_cols:
                keep_cols.append(col)

        self.data = (
            self.df[keep_cols]
            .copy()
            .sort_values("date")
            .reset_index(drop=True)
        )

        print("\n" + "=" * 70)
        print(f"LOCAL PROJECTIONS{' (' + self.name + ')' if self.name else ''}")
        print("=" * 70)
        print(f"\nShock: {self.shock_var}")
        print(f"Respuestas: {self.response_vars}")
        print(f"Controles: {self.control_vars}")
        print(f"Lags de controles: {self.lags}")
        print(f"Horizonte máximo: {self.horizons}")
        print(f"Observaciones iniciales: {len(self.data)}")
        print(
            f"Periodo: {self.data['date'].iloc[0].strftime('%Y-%m')} a "
            f"{self.data['date'].iloc[-1].strftime('%Y-%m')}"
        )

    def _build_lp_dataset(self, response_var: str, h: int) -> pd.DataFrame:
        """
        Construye dataset para horizonte h:
            response_{t+h} sobre shock_t y controles rezagados.
        """
        df_lp = self.data.copy()

        # Variable dependiente futura
        df_lp[f"{response_var}_lead_{h}"] = df_lp[response_var].shift(-h)

        # Controles rezagados
        regressors = [self.shock_var]

        for lag in range(1, self.lags + 1):
            col = f"{self.shock_var}_lag{lag}"
            df_lp[col] = df_lp[self.shock_var].shift(lag)
            regressors.append(col)

        for var in self.control_vars:
            for lag in range(1, self.lags + 1):
                col = f"{var}_lag{lag}"
                df_lp[col] = df_lp[var].shift(lag)
                regressors.append(col)

        use_cols = ["date", f"{response_var}_lead_{h}"] + regressors
        df_lp = df_lp[use_cols].dropna().copy()

        return df_lp

    def estimate_single_lp(self, response_var: str, h: int) -> dict:
        """
        Estima una sola regresión LP para una variable y un horizonte.
        Usa errores HAC(Newey-West) con maxlags = h+1.
        """
        df_lp = self._build_lp_dataset(response_var, h)

        y = df_lp[f"{response_var}_lead_{h}"]
        X = df_lp.drop(columns=["date", f"{response_var}_lead_{h}"])
        X = sm.add_constant(X)

        model = sm.OLS(y, X)
        # HAC robusto a autocorrelación por horizontes solapados
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": max(1, h + 1)})

        beta = res.params[self.shock_var]
        se = res.bse[self.shock_var]

        return {
            "response_var": response_var,
            "horizon": h,
            "nobs": int(res.nobs),
            "beta": beta,
            "se": se,
            "tstat": beta / se if se != 0 else np.nan,
            "pvalue": res.pvalues[self.shock_var],
            "ci_lower_90": beta - 1.645 * se,
            "ci_upper_90": beta + 1.645 * se,
            "ci_lower_95": beta - 1.96 * se,
            "ci_upper_95": beta + 1.96 * se,
            "r2": res.rsquared,
        }

    def estimate_irf(self, response_var: str) -> pd.DataFrame:
        """
        Estima todos los horizontes para una variable respuesta.
        """
        print("\n" + "-" * 70)
        print(f"Estimando LP: shock {self.shock_var} → {response_var}")
        print("-" * 70)

        rows = []
        for h in range(self.horizons + 1):
            out = self.estimate_single_lp(response_var, h)
            rows.append(out)

            if h in [0, 1, 3, 6, 12, 24]:
                print(
                    f"h={h:2d} | beta={out['beta']:+.6f} | "
                    f"se={out['se']:.6f} | p={out['pvalue']:.4f}"
                )

        df_res = pd.DataFrame(rows)
        self.results[response_var] = df_res
        return df_res

    def estimate_all(self):
        """
        Estima LP para todas las variables respuesta.
        """
        for response_var in self.response_vars:
            self.estimate_irf(response_var)

        self._save_results()
        self.plot_all_irfs()

    def _save_results(self):
        """
        Guarda tablas de resultados.
        """
        suffix = f"_{self.name}" if self.name else ""

        for response_var, df_res in self.results.items():
            out_path = RESULTS_TABLES / f"lp_{self.shock_var}_to_{response_var}{suffix}.csv"
            df_res.to_csv(out_path, index=False)
            print(f"\n✓ Resultados guardados: {out_path}")

    def plot_irf(self, response_var: str, conf: str = "95"):
        """
        Gráfico individual de IRF.
        """
        if response_var not in self.results:
            raise ValueError(f"No hay resultados para {response_var}. Ejecuta estimate_irf primero.")

        df_res = self.results[response_var]

        if conf == "90":
            low = "ci_lower_90"
            high = "ci_upper_90"
            label = "90%"
        else:
            low = "ci_lower_95"
            high = "ci_upper_95"
            label = "95%"

        plt.figure(figsize=(10, 6))
        plt.plot(df_res["horizon"], df_res["beta"], lw=2, label="IRF (LP)")
        plt.fill_between(
            df_res["horizon"],
            df_res[low],
            df_res[high],
            alpha=0.25,
            label=f"IC {label}"
        )
        plt.axhline(0, color="black", linestyle="--", linewidth=1)
        plt.xlabel("Meses después del shock")
        plt.ylabel("Respuesta")
        plt.title(f"Local Projection: {response_var} a shock en {self.shock_var}")
        plt.legend()
        plt.tight_layout()

        suffix = f"_{self.name}" if self.name else ""
        out_path = RESULTS_FIGURES / f"lp_{self.shock_var}_to_{response_var}{suffix}.png"
        plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()

        print(f"✓ Gráfico guardado: {out_path}")

    def plot_all_irfs(self, conf: str = "95"):
        """
        Gráfico conjunto para todas las respuestas.
        """
        n = len(self.response_vars)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=True)

        if n == 1:
            axes = [axes]

        for ax, response_var in zip(axes, self.response_vars):
            df_res = self.results[response_var]

            if conf == "90":
                low = "ci_lower_90"
                high = "ci_upper_90"
                label = "90%"
            else:
                low = "ci_lower_95"
                high = "ci_upper_95"
                label = "95%"

            ax.plot(df_res["horizon"], df_res["beta"], lw=2)
            ax.fill_between(
                df_res["horizon"],
                df_res[low],
                df_res[high],
                alpha=0.25
            )
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_title(f"{self.shock_var} → {response_var}")
            ax.set_ylabel("Respuesta")

        axes[-1].set_xlabel("Meses después del shock")
        plt.suptitle(f"Local Projections{' - ' + self.name if self.name else ''}", y=0.995)
        plt.tight_layout()

        suffix = f"_{self.name}" if self.name else ""
        out_path = RESULTS_FIGURES / f"lp_all_{self.shock_var}{suffix}.png"
        plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close()

        print(f"✓ Gráfico conjunto guardado: {out_path}")


def main():
    print("\n" + "=" * 70)
    print(" LOCAL PROJECTIONS - SHOCKS DE LIQUIDEZ")
    print("=" * 70)

    start_time = datetime.now()

    print("\nCargando datos...")
    data_path = DATA_PROCESSED / "monthly_data.csv"
    df = pd.read_csv(data_path, parse_dates=["date"])

    # -----------------------------------------------------------------
    # ESPECIFICACIÓN BASE: RESERVAS
    # -----------------------------------------------------------------
    shock_var = "growth_total_reserves"

    response_vars = [
        "ret_sp500",
        "delta_spread",
        "delta_vix"
    ]

    control_vars = [
        "delta_spread",
        "delta_vix",
        "ret_sp500"
    ]

    lp = LocalProjectionAnalysis(
        df=df,
        shock_var=shock_var,
        response_vars=response_vars,
        control_vars=control_vars,
        lags=6,
        horizons=24,
        name="reserves"
    )

    lp.estimate_all()

    # Gráficos individuales extra
    for y in response_vars:
        lp.plot_irf(y, conf="95")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱ Tiempo total: {elapsed:.1f}s")
    print("\n✓ Pipeline LP completado\n")


if __name__ == "__main__":
    main()