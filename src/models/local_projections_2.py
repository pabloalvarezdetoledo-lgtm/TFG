"""
Local Projections with regime dependence
Jordà (2005) style impulse responses with interaction by regime.

Modelo base:
    y_{t+h} = alpha_h + beta1_h * shock_t
                    + beta2_h * (shock_t * regime_t)
                    + sum_j Gamma_h,j' X_{t-j}
                    + epsilon_{t+h}

Interpretación:
    beta_normal = beta1_h
    beta_high   = beta1_h + beta2_h

Este script:
1. Carga monthly_data.csv
2. Construye regímenes de VIX:
   - high_vix_20   : VIX > 20
   - high_vix_q75  : VIX > percentil 75
3. Estima LP con interacción por régimen
4. Guarda tablas y gráficos

Guárdalo como:
    src/models/local_projections_regime.py
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import DATA_PROCESSED, RESULTS_TABLES, RESULTS_FIGURES, FIGURE_DPI

warnings.filterwarnings("ignore")


def local_projections_regime(
    df: pd.DataFrame,
    shock_var: str,
    response_var: str,
    regime_var: str,
    control_vars: list[str],
    max_horizon: int = 24,
    n_lags: int = 6,
) -> pd.DataFrame:
    """
    Estima Local Projections con interacción por régimen.

    Parámetros
    ----------
    df : DataFrame
        Debe contener shock_var, response_var, regime_var y control_vars.
    shock_var : str
        Variable shock, por ejemplo 'growth_total_reserves'
    response_var : str
        Variable respuesta, por ejemplo 'ret_sp500'
    regime_var : str
        Dummy de régimen, por ejemplo 'high_vix_20'
    control_vars : list[str]
        Variables a incluir con rezagos.
        Recomendado: ['delta_spread', 'delta_vix', 'ret_sp500']
        o añadir 'delta_2y_yield' como robustez.
    max_horizon : int
        Horizonte máximo h
    n_lags : int
        Número de rezagos de controles y del shock

    Retorna
    -------
    DataFrame con betas, errores estándar, p-values y bandas.
    """
    needed = [shock_var, response_var, regime_var] + control_vars
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    results = []

    for h in range(max_horizon + 1):
        work = df.copy()

        # Dependiente futura
        y_name = f"{response_var}_lead_{h}"
        work[y_name] = work[response_var].shift(-h)

        # Shock contemporáneo e interacción
        work["shock"] = work[shock_var]
        work["shock_regime"] = work[shock_var] * work[regime_var]

        regressors = ["shock", "shock_regime"]

        # Lags del propio shock
        for lag in range(1, n_lags + 1):
            col = f"{shock_var}_L{lag}"
            work[col] = work[shock_var].shift(lag)
            regressors.append(col)

        # Lags de controles
        for var in control_vars:
            for lag in range(1, n_lags + 1):
                col = f"{var}_L{lag}"
                work[col] = work[var].shift(lag)
                regressors.append(col)

        # Dataset final
        use_cols = [y_name] + regressors
        sample = work[use_cols].dropna().copy()

        y = sample[y_name]
        X = sample[regressors]
        X = sm.add_constant(X)

        model = sm.OLS(y, X)
        # HAC / Newey-West por horizontes solapados
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": max(1, h + 1)})

        beta1 = res.params["shock"]
        beta2 = res.params["shock_regime"]

        se1 = res.bse["shock"]
        se2 = res.bse["shock_regime"]

        # Efecto total en régimen alto
        beta_high = beta1 + beta2

        # Delta method para se(beta1 + beta2)
        vcov = res.cov_params().loc[["shock", "shock_regime"], ["shock", "shock_regime"]]
        se_high = np.sqrt(
            vcov.loc["shock", "shock"]
            + vcov.loc["shock_regime", "shock_regime"]
            + 2 * vcov.loc["shock", "shock_regime"]
        )

        # t y p-value del efecto total en régimen alto
        t_high = beta_high / se_high if se_high > 0 else np.nan
        p_high = 2 * (1 - norm.cdf(np.abs(t_high))) if np.isfinite(t_high) else np.nan

        results.append(
            {
                "horizon": h,
                "beta_normal": beta1,
                "se_normal": se1,
                "t_normal": res.tvalues["shock"],
                "p_normal": res.pvalues["shock"],
                "beta_interaction": beta2,
                "se_interaction": se2,
                "t_interaction": res.tvalues["shock_regime"],
                "p_interaction": res.pvalues["shock_regime"],
                "beta_high": beta_high,
                "se_high": se_high,
                "t_high": t_high,
                "p_high": p_high,
                "ci95_low_normal": beta1 - 1.96 * se1,
                "ci95_high_normal": beta1 + 1.96 * se1,
                "ci95_low_high": beta_high - 1.96 * se_high,
                "ci95_high_high": beta_high + 1.96 * se_high,
                "r2": res.rsquared,
                "n_obs": int(res.nobs),
            }
        )

    return pd.DataFrame(results)


def save_results_table(df_res: pd.DataFrame, shock_var: str, response_var: str, regime_var: str, name: str):
    filename = f"lp_regime_{shock_var}_to_{response_var}_{regime_var}_{name}.csv"
    out_path = RESULTS_TABLES / filename
    df_res.to_csv(out_path, index=False)
    print(f"✓ Tabla guardada: {out_path}")


def plot_lp_regime(df_res: pd.DataFrame, shock_var: str, response_var: str, regime_var: str, name: str):
    plt.figure(figsize=(10, 6))

    # Régimen normal
    plt.plot(
        df_res["horizon"],
        df_res["beta_normal"],
        linewidth=2,
        label="Régimen normal",
    )
    plt.fill_between(
        df_res["horizon"],
        df_res["ci95_low_normal"],
        df_res["ci95_high_normal"],
        alpha=0.20,
    )

    # Régimen alto
    plt.plot(
        df_res["horizon"],
        df_res["beta_high"],
        linewidth=2,
        label="Régimen alto",
    )
    plt.fill_between(
        df_res["horizon"],
        df_res["ci95_low_high"],
        df_res["ci95_high_high"],
        alpha=0.20,
    )

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Horizonte (meses)")
    plt.ylabel("Respuesta estimada")
    plt.title(f"LP con régimen: {shock_var} → {response_var} ({regime_var})")
    plt.legend()
    plt.tight_layout()

    filename = f"lp_regime_{shock_var}_to_{response_var}_{regime_var}_{name}.png"
    out_path = RESULTS_FIGURES / filename
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"✓ Gráfico guardado: {out_path}")


def print_key_results(df_res: pd.DataFrame, response_var: str):
    print("\n" + "-" * 70)
    print(f"Resultados clave: {response_var}")
    print("-" * 70)
    for h in [0, 1, 3, 6, 12, 24]:
        if h <= df_res["horizon"].max():
            row = df_res.loc[df_res["horizon"] == h].iloc[0]
            print(
                f"h={h:2d} | "
                f"normal={row['beta_normal']:+.6f} (p={row['p_normal']:.4f}) | "
                f"high={row['beta_high']:+.6f} (p={row['p_high']:.4f}) | "
                f"interaction={row['beta_interaction']:+.6f} (p={row['p_interaction']:.4f})"
            )


def main():
    print("\n" + "=" * 70)
    print(" LOCAL PROJECTIONS CON DEPENDENCIA DE RÉGIMEN")
    print("=" * 70)

    start_time = datetime.now()

    # -----------------------------------------------------------------
    # Cargar datos
    # -----------------------------------------------------------------
    print("\nCargando datos...")
    data_path = DATA_PROCESSED / "monthly_data.csv"
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # -----------------------------------------------------------------
    # Regímenes
    # -----------------------------------------------------------------
    # Umbral económico
    df["high_vix_20"] = (df["vix"] > 20).astype(int)

    # Umbral por percentil 75
    vix_q75 = df["vix"].quantile(0.75)
    df["high_vix_q75"] = (df["vix"] > vix_q75).astype(int)

    # Control de expectativas / robustez
    if "treasury_2y" in df.columns:
        df["delta_2y_yield"] = df["treasury_2y"].diff()

    print(f"Observaciones: {len(df)}")
    print(f"Periodo: {df['date'].iloc[0].strftime('%Y-%m')} a {df['date'].iloc[-1].strftime('%Y-%m')}")
    print(f"Régimen high_vix_20: {df['high_vix_20'].mean():.1%}")
    print(f"Régimen high_vix_q75: {df['high_vix_q75'].mean():.1%}")
    print(f"Percentil 75 del VIX: {vix_q75:.2f}")

    # -----------------------------------------------------------------
    # Especificación base
    # -----------------------------------------------------------------
    shock_var = "growth_total_reserves"
    response_vars = ["ret_sp500", "delta_spread", "delta_vix"]

    # Modelo base
    control_vars_base = ["delta_spread", "delta_vix", "ret_sp500"]

    # Robustez con expectativas
    control_vars_plus_2y = control_vars_base.copy()
    if "delta_2y_yield" in df.columns:
        control_vars_plus_2y.append("delta_2y_yield")

    # -----------------------------------------------------------------
    # CORRIDA 1: régimen VIX > 20, especificación base
    # -----------------------------------------------------------------
    regime_var = "high_vix_20"
    name = "base"

    print("\n" + "=" * 70)
    print("MODELO 1: Régimen VIX > 20, especificación base")
    print("=" * 70)

    for response_var in response_vars:
        df_res = local_projections_regime(
            df=df,
            shock_var=shock_var,
            response_var=response_var,
            regime_var=regime_var,
            control_vars=control_vars_base,
            max_horizon=24,
            n_lags=6,
        )
        print_key_results(df_res, response_var)
        save_results_table(df_res, shock_var, response_var, regime_var, name)
        plot_lp_regime(df_res, shock_var, response_var, regime_var, name)

    # -----------------------------------------------------------------
    # CORRIDA 2: régimen VIX > 20, robustez con 2y yield
    # -----------------------------------------------------------------
    if "delta_2y_yield" in df.columns:
        name = "plus2y"

        print("\n" + "=" * 70)
        print("MODELO 2: Régimen VIX > 20, con control delta_2y_yield")
        print("=" * 70)

        for response_var in response_vars:
            df_res = local_projections_regime(
                df=df,
                shock_var=shock_var,
                response_var=response_var,
                regime_var=regime_var,
                control_vars=control_vars_plus_2y,
                max_horizon=24,
                n_lags=6,
            )
            print_key_results(df_res, response_var)
            save_results_table(df_res, shock_var, response_var, regime_var, name)
            plot_lp_regime(df_res, shock_var, response_var, regime_var, name)

    # -----------------------------------------------------------------
    # CORRIDA 3: robustez con percentil 75
    # -----------------------------------------------------------------
    regime_var = "high_vix_q75"
    name = "q75"

    print("\n" + "=" * 70)
    print("MODELO 3: Régimen VIX > p75, especificación base")
    print("=" * 70)

    for response_var in response_vars:
        df_res = local_projections_regime(
            df=df,
            shock_var=shock_var,
            response_var=response_var,
            regime_var=regime_var,
            control_vars=control_vars_base,
            max_horizon=24,
            n_lags=6,
        )
        print_key_results(df_res, response_var)
        save_results_table(df_res, shock_var, response_var, regime_var, name)
        plot_lp_regime(df_res, shock_var, response_var, regime_var, name)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n⏱ Tiempo total: {elapsed:.1f}s")
    print("\n✓ Pipeline LP con régimen completado\n")


if __name__ == "__main__":
    main()