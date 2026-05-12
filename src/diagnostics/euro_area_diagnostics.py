"""
Diagnosticos econometricos iniciales para la especificacion euro area.

Este script prepara tablas descriptivas, tests de estacionariedad y una
inspeccion lead-lag. No estima modelos finales ni interpreta causalidad.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


# Anadir src/ al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import DATA_PROCESSED, RESULTS_TABLES, PROJECT_ROOT


DATA_PATH = DATA_PROCESSED / "monthly_data_euro_area.csv"

BASELINE_VARIABLES = [
    "eurostoxx50",
    "log_eurostoxx50",
    "ret_eurostoxx50",
    "ciss",
    "delta_ciss",
    "deposit_facility_rate",
    "delta_deposit_rate",
    "eurosystem_total_assets",
    "log_eurosystem_total_assets",
    "growth_eurosystem_total_assets",
]

SHORT_SAMPLE_VARIABLES = [
    "estr",
    "euro_area_2y_yield",
    "euro_area_10y_yield",
    "euro_area_slope_curve",
    "delta_euro_area_slope",
    "excess_liquidity",
]

UNRESOLVED_VARIABLES = [
    "vstoxx",
    "emu_growth",
    "emu_value",
    "emu_value_minus_growth",
    "app_holdings",
    "pepp_holdings",
    "european_credit_spread",
]

STATIONARITY_VARIABLES = [
    "log_eurostoxx50",
    "ret_eurostoxx50",
    "ciss",
    "delta_ciss",
    "deposit_facility_rate",
    "delta_deposit_rate",
    "log_eurosystem_total_assets",
    "growth_eurosystem_total_assets",
    "euro_area_slope_curve",
    "delta_euro_area_slope",
]

CORRELATION_VARIABLES = [
    "growth_eurosystem_total_assets",
    "ret_eurostoxx50",
    "delta_ciss",
    "ciss",
    "delta_deposit_rate",
    "euro_area_slope_curve",
    "delta_euro_area_slope",
]

LEAD_LAG_TARGETS = [
    "ret_eurostoxx50",
    "delta_ciss",
    "delta_deposit_rate",
    "ciss",
]


def load_euro_area_data():
    """Carga el dataset mensual euro area."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {DATA_PATH}. Ejecuta primero src/data_processing/create_monthly.py"
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"OK datos euro area cargados: {len(df)} observaciones")
    return df


def all_relevant_variables():
    """Devuelve las variables relevantes para diagnostico."""
    return BASELINE_VARIABLES + SHORT_SAMPLE_VARIABLES + UNRESOLVED_VARIABLES


def valid_dates(df, variable):
    """Devuelve primera y ultima fecha valida de una variable."""
    if variable not in df.columns:
        return "", ""

    valid = df.loc[df[variable].notna(), "date"]
    if valid.empty:
        return "", ""

    return valid.min().strftime("%Y-%m-%d"), valid.max().strftime("%Y-%m-%d")


def variable_group(variable):
    """Clasifica una variable segun su papel empirico."""
    if variable in BASELINE_VARIABLES:
        return "baseline"
    if variable in SHORT_SAMPLE_VARIABLES:
        return "short_sample"
    if variable in UNRESOLVED_VARIABLES:
        return "unresolved"
    return "other"


def compute_missing_values(df, variables):
    """Calcula cobertura y missing values por variable."""
    rows = []

    for variable in variables:
        exists = variable in df.columns
        if exists:
            missing = int(df[variable].isna().sum())
            non_missing = int(df[variable].notna().sum())
            first_valid, last_valid = valid_dates(df, variable)
        else:
            missing = int(len(df))
            non_missing = 0
            first_valid, last_valid = "", ""

        rows.append({
            "variable": variable,
            "group": variable_group(variable),
            "exists": exists,
            "observations": non_missing,
            "first_valid_date": first_valid,
            "last_valid_date": last_valid,
            "missing_values": missing,
            "missing_pct": round(missing / len(df) * 100, 2) if len(df) else np.nan,
        })

    return pd.DataFrame(rows)


def compute_descriptives(df, variables):
    """Calcula estadisticos descriptivos para variables relevantes."""
    rows = []

    for variable in variables:
        exists = variable in df.columns
        series = pd.to_numeric(df[variable], errors="coerce").dropna() if exists else pd.Series(dtype=float)
        first_valid, last_valid = valid_dates(df, variable)
        missing = int(df[variable].isna().sum()) if exists else int(len(df))

        rows.append({
            "variable": variable,
            "group": variable_group(variable),
            "exists": exists,
            "observations": int(series.shape[0]),
            "first_valid_date": first_valid,
            "last_valid_date": last_valid,
            "missing_values": missing,
            "mean": series.mean() if len(series) else np.nan,
            "std": series.std(ddof=1) if len(series) > 1 else np.nan,
            "min": series.min() if len(series) else np.nan,
            "max": series.max() if len(series) else np.nan,
            "skewness": series.skew() if len(series) > 2 else np.nan,
            "kurtosis": series.kurt() if len(series) > 3 else np.nan,
        })

    return pd.DataFrame(rows)


def interpret_stationarity(adf_pvalue, kpss_pvalue):
    """
    Interpreta ADF y KPSS sin forzar conclusiones.

    ADF H0: raiz unitaria. KPSS H0: estacionariedad.
    """
    adf_rejects = adf_pvalue < 0.05 if pd.notna(adf_pvalue) else False
    kpss_rejects = kpss_pvalue < 0.05 if pd.notna(kpss_pvalue) else False

    if adf_rejects and not kpss_rejects:
        return "evidencia_estacionaria"
    if not adf_rejects and kpss_rejects:
        return "evidencia_no_estacionaria"
    if adf_rejects and kpss_rejects:
        return "ambigua_adf_kpss_discrepan"
    return "ambigua_no_concluyente"


def run_stationarity_tests(df, variables, min_obs=36):
    """Ejecuta ADF y KPSS para las variables disponibles."""
    rows = []

    for variable in variables:
        if variable not in df.columns:
            print(f"AVISO: {variable} no existe; se omite en tests de estacionariedad")
            rows.append({
                "variable": variable,
                "observations": 0,
                "adf_stat": np.nan,
                "adf_pvalue": np.nan,
                "kpss_stat": np.nan,
                "kpss_pvalue": np.nan,
                "interpretation": "variable_no_disponible",
                "notes": "Variable no encontrada en monthly_data_euro_area.csv",
            })
            continue

        series = pd.to_numeric(df[variable], errors="coerce").dropna()
        if len(series) < min_obs:
            print(f"AVISO: {variable} tiene solo {len(series)} observaciones; test no fiable")
            rows.append({
                "variable": variable,
                "observations": int(len(series)),
                "adf_stat": np.nan,
                "adf_pvalue": np.nan,
                "kpss_stat": np.nan,
                "kpss_pvalue": np.nan,
                "interpretation": "muestra_insuficiente",
                "notes": f"Menos de {min_obs} observaciones validas",
            })
            continue

        notes = ""
        try:
            adf_result = adfuller(series, autolag="AIC")
            adf_stat = adf_result[0]
            adf_pvalue = adf_result[1]
        except Exception as exc:
            adf_stat = np.nan
            adf_pvalue = np.nan
            notes += f"ADF fallo: {exc}. "

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_result = kpss(series, regression="c", nlags="auto")
            kpss_stat = kpss_result[0]
            kpss_pvalue = kpss_result[1]
        except Exception as exc:
            kpss_stat = np.nan
            kpss_pvalue = np.nan
            notes += f"KPSS fallo: {exc}. "

        rows.append({
            "variable": variable,
            "observations": int(len(series)),
            "adf_stat": adf_stat,
            "adf_pvalue": adf_pvalue,
            "kpss_stat": kpss_stat,
            "kpss_pvalue": kpss_pvalue,
            "interpretation": interpret_stationarity(adf_pvalue, kpss_pvalue),
            "notes": notes.strip(),
        })

    return pd.DataFrame(rows)


def compute_correlation_matrix(df):
    """Calcula matriz de correlaciones para variables disponibles."""
    available = [var for var in CORRELATION_VARIABLES if var in df.columns and df[var].notna().sum() >= 12]
    if len(available) < 2:
        print("AVISO: no hay suficientes variables para matriz de correlacion")
        return pd.DataFrame()

    return df[available].corr()


def compute_lead_lag_correlations(df, max_lag=12):
    """
    Calcula correlaciones lead-lag descriptivas.

    Lag negativo: la variable financiera lidera a la liquidez.
    Lag positivo: la liquidez lidera a la variable financiera.
    """
    liquidity = "growth_eurosystem_total_assets"
    rows = []

    if liquidity not in df.columns:
        print("AVISO: falta growth_eurosystem_total_assets para lead-lag")
        return pd.DataFrame()

    for target in LEAD_LAG_TARGETS:
        if target not in df.columns:
            print(f"AVISO: falta {target} para lead-lag")
            continue

        for lag in range(-max_lag, max_lag + 1):
            shifted_target = df[target].shift(-lag)
            pair = pd.DataFrame({
                liquidity: df[liquidity],
                target: shifted_target,
            }).dropna()
            corr = pair[liquidity].corr(pair[target]) if len(pair) >= 24 else np.nan

            rows.append({
                "liquidity_variable": liquidity,
                "financial_variable": target,
                "lag_months": lag,
                "interpretation": (
                    "financial_variable_leads_liquidity"
                    if lag < 0 else
                    "liquidity_leads_financial_variable"
                    if lag > 0 else
                    "contemporaneous"
                ),
                "correlation": corr,
                "observations": int(len(pair)),
            })

    return pd.DataFrame(rows)


def baseline_readiness_flags(df):
    """Evalua si el dataset soporta las especificaciones iniciales."""
    required = [
        "growth_eurosystem_total_assets",
        "ret_eurostoxx50",
        "delta_ciss",
        "delta_deposit_rate",
    ]
    ready = all(var in df.columns and df[var].dropna().shape[0] >= 250 for var in required)
    return {
        "baseline_ready": ready,
        "yield_curve_ready": "euro_area_slope_curve" in df.columns and df["euro_area_slope_curve"].dropna().shape[0] >= 200,
        "excess_liquidity_full_sample_ready": "growth_excess_liquidity" in df.columns and df["growth_excess_liquidity"].dropna().shape[0] >= 250,
        "growth_value_ready": "emu_value_minus_growth" in df.columns and df["emu_value_minus_growth"].dropna().shape[0] >= 120,
        "vstoxx_ready": "delta_vstoxx" in df.columns and df["delta_vstoxx"].dropna().shape[0] >= 120,
    }


def write_model_readiness_report(df, missing_df, stationarity_df, lead_lag_df):
    """Escribe el reporte de preparacion de modelos euro area."""
    output_path = PROJECT_ROOT / "EURO_AREA_MODEL_READINESS.md"
    flags = baseline_readiness_flags(df)

    sample_start = df["date"].min().strftime("%Y-%m-%d")
    sample_end = df["date"].max().strftime("%Y-%m-%d")
    nobs = len(df)

    missing_lookup = missing_df.set_index("variable")
    stationarity_lines = [
        f"- `{row.variable}`: {row.interpretation} "
        f"(ADF p={row.adf_pvalue:.4f}, KPSS p={row.kpss_pvalue:.4f})"
        if pd.notna(row.adf_pvalue) and pd.notna(row.kpss_pvalue)
        else f"- `{row.variable}`: {row.interpretation}"
        for row in stationarity_df.itertuples()
    ]

    lead_lag_summary = []
    if not lead_lag_df.empty:
        for target in LEAD_LAG_TARGETS:
            sub = lead_lag_df[lead_lag_df["financial_variable"] == target].dropna(subset=["correlation"])
            if sub.empty:
                continue
            best = sub.iloc[sub["correlation"].abs().argmax()]
            lead_lag_summary.append(
                f"- `{target}`: mayor correlacion absoluta en lag {int(best['lag_months'])} "
                f"({best['correlation']:.3f}); uso descriptivo, no causal."
            )

    content = [
        "# Preparacion de modelos euro area - Fase 3",
        "",
        "## Estado del dataset",
        "",
        f"- Dataset usado: `data/processed/monthly_data_euro_area.csv`",
        f"- Muestra mensual: `{sample_start}` a `{sample_end}`",
        f"- Observaciones mensuales: `{nobs}`",
        "- Objeto empirico: area del euro y ECB/Eurosistema.",
        "- Variables US: solo legado o benchmark; no entran en esta preparacion.",
        "",
        "## Decision central de liquidez",
        "",
        "- Proxy baseline full-sample: `growth_eurosystem_total_assets`.",
        "- Motivo: `eurosystem_total_assets` cubre 2000-2026 sin missing mensuales.",
        "- `excess_liquidity` queda como robustez/descriptiva de muestra corta porque la serie directa empieza en 2024-09.",
        "- No se traslada mecanicamente la formula US `WALCL - RRP - TGA`.",
        "",
        "## Nota sobre EURO STOXX 50",
        "",
        "- El baseline `eurostoxx50` viene del ECB Data Portal (`RTD.M.S0.N.C_DJE50.X`).",
        "- Esa fuente es una serie mensual de promedio de observaciones del periodo, no cierre de fin de mes.",
        "- Yahoo `^STOXX50E` queda como robustez de muestra corta desde 2007, no como baseline 2000-2026.",
        "",
        "## Cobertura de variables clave",
        "",
    ]

    for variable in BASELINE_VARIABLES + SHORT_SAMPLE_VARIABLES + UNRESOLVED_VARIABLES:
        if variable in missing_lookup.index:
            row = missing_lookup.loc[variable]
            content.append(
                f"- `{variable}`: obs={int(row['observations'])}, "
                f"missing={int(row['missing_values'])}, "
                f"primer valido `{row['first_valid_date'] or 'n/a'}`, "
                f"ultimo valido `{row['last_valid_date'] or 'n/a'}`."
            )

    content.extend([
        "",
        "## Diagnosticos de estacionariedad",
        "",
        "Los tests ADF y KPSS se usan solo como guia de especificacion. Si discrepan, se reporta ambiguedad.",
        "",
        *(stationarity_lines or ["- No se generaron tests de estacionariedad."]),
        "",
        "## Inspeccion lead-lag",
        "",
        "Convencion: lag negativo significa que la variable financiera lidera a la liquidez; lag positivo significa que la liquidez lidera a la variable financiera.",
        "",
        *(lead_lag_summary or ["- No se generaron correlaciones lead-lag suficientes."]),
        "",
        "## Recomendacion para Fase 4",
        "",
        "Primera funcion de reaccion recomendada:",
        "",
        "```text",
        "growth_eurosystem_total_assets_t = alpha",
        "  + sum beta_i growth_eurosystem_total_assets_{t-i}",
        "  + sum gamma_i ciss_{t-i}",
        "  + sum delta_i ret_eurostoxx50_{t-i}",
        "  + sum theta_i delta_deposit_rate_{t-i}",
        "  + error_t",
        "```",
        "",
        "Primera especificacion Local Projection recomendada:",
        "",
        "```text",
        "y_{t+h} - y_{t-1} = alpha_h + beta_h shock_liquidity_t + controls_t + error_{t+h}",
        "```",
        "",
        "Respuestas candidatas: `ret_eurostoxx50`, `delta_ciss`, `ciss`, `delta_deposit_rate`.",
        "",
        "## Estado de readiness",
        "",
        f"- Baseline Phase 4 listo: `{'si' if flags['baseline_ready'] else 'no'}`",
        f"- Curva 2Y/10Y lista como extension desde 2004: `{'si' if flags['yield_curve_ready'] else 'no'}`",
        f"- Exceso de liquidez listo para full-sample: `{'si' if flags['excess_liquidity_full_sample_ready'] else 'no'}`",
        f"- Growth/Value listo: `{'si' if flags['growth_value_ready'] else 'no'}`",
        f"- VSTOXX listo: `{'si' if flags['vstoxx_ready'] else 'no'}`",
        "",
        "## Advertencias metodologicas",
        "",
        "- No se debe interpretar la correlacion lead-lag como causalidad.",
        "- No se deben diferenciar variables mecanicamente si se pierde la interpretacion teorica.",
        "- Los modelos definitivos deben controlar la endogeneidad temporal y reportar robustez de rezagos.",
        "- VSTOXX, Growth/Value, APP/PEPP y spreads corporativos siguen no disponibles o no validados.",
    ])

    output_path.write_text("\n".join(content) + "\n", encoding="utf-8")
    print(f"OK reporte de readiness guardado: {output_path}")
    return output_path


def main():
    """Ejecuta diagnosticos de Fase 3."""
    print("\n" + "=" * 70)
    print("DIAGNOSTICOS EURO AREA - FASE 3")
    print("=" * 70)

    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    df = load_euro_area_data()
    variables = all_relevant_variables()

    missing_df = compute_missing_values(df, variables)
    missing_path = RESULTS_TABLES / "euro_area_missing_values.csv"
    missing_df.to_csv(missing_path, index=False)
    print(f"OK missing values: {missing_path}")

    descriptives_df = compute_descriptives(df, variables)
    descriptives_path = RESULTS_TABLES / "euro_area_descriptives.csv"
    descriptives_df.to_csv(descriptives_path, index=False)
    print(f"OK descriptivos: {descriptives_path}")

    stationarity_df = run_stationarity_tests(df, STATIONARITY_VARIABLES)
    stationarity_path = RESULTS_TABLES / "euro_area_stationarity_tests.csv"
    stationarity_df.to_csv(stationarity_path, index=False)
    print(f"OK estacionariedad: {stationarity_path}")

    corr_df = compute_correlation_matrix(df)
    corr_path = RESULTS_TABLES / "euro_area_correlation_matrix.csv"
    corr_df.to_csv(corr_path)
    print(f"OK matriz de correlacion: {corr_path}")

    lead_lag_df = compute_lead_lag_correlations(df)
    lead_lag_path = RESULTS_TABLES / "euro_area_lead_lag_correlations.csv"
    lead_lag_df.to_csv(lead_lag_path, index=False)
    print(f"OK lead-lag: {lead_lag_path}")

    write_model_readiness_report(df, missing_df, stationarity_df, lead_lag_df)

    print("\nOK diagnosticos euro area completados")


if __name__ == "__main__":
    main()
