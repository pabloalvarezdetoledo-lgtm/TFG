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
    RESULTS_TABLES,
    PROJECT_ROOT,
    START_DATE,
    END_DATE,
    MONTHLY_FREQ,
    REGION,
    EURO_AREA_MONTHLY_INPUTS,
    EURO_AREA_DERIVED_VARIABLES,
    EURO_AREA_PLACEHOLDER_SERIES
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


def warn_missing_euro_area_csv(filepath, value_col, required):
    """Avisa de CSV euro area ausente sin interrumpir el pipeline."""
    priority = "REQUERIDO" if required else "opcional"
    print(
        f"  AVISO {priority}: no se encontro CSV para {value_col}: {filepath}. "
        "Se continua sin esta serie."
    )


def load_euro_area_data():
    """
    Carga las series euro area configuradas y las resamplea a mensual.

    Las fuentes pendientes se esperan como CSV con columna date y una columna
    con el mismo nombre que la variable. Si faltan, se imprime aviso y se sigue.
    """
    print("\n" + "="*70)
    print("CARGANDO Y RESAMPLING A MENSUAL - EURO AREA")
    print("="*70)

    data = {}

    for value_col, meta in EURO_AREA_MONTHLY_INPUTS.items():
        filepath = DATA_RAW / meta["filename"]
        method = meta.get("method", "last")
        required = meta.get("required", False)
        status = "requerida" if required else "opcional"

        print(f"\n- {value_col} ({status}, metodo={method})...")

        if not filepath.exists():
            warn_missing_euro_area_csv(filepath, value_col, required)
            data[value_col] = None
            continue

        try:
            data[value_col] = load_and_resample(filepath, value_col, method=method)
            print(f"  OK {value_col}: {len(data[value_col])} meses")
        except Exception as exc:
            print(f"  AVISO: no se pudo cargar {value_col} desde {filepath}: {exc}")
            data[value_col] = None

    return data


def merge_euro_area_series(data):
    """
    Hace merge de series euro area sin exigir que todos los CSV existan.

    EURO STOXX 50 es la base preferente. Si aun no existe, se usa la primera
    serie disponible para permitir pruebas parciales del pipeline.
    """
    print("\n" + "="*70)
    print("MERGE DE SERIES EURO AREA A FRECUENCIA MENSUAL")
    print("="*70)

    for key in data.keys():
        if data[key] is not None:
            data[key]['date'] = pd.to_datetime(data[key]['date'])
            data[key]['date'] = data[key]['date'] + pd.offsets.MonthEnd(0)

    base_name = "eurostoxx50" if data.get("eurostoxx50") is not None else None

    if base_name is None:
        for key, value in data.items():
            if value is not None:
                base_name = key
                print(
                    f"  AVISO: eurostoxx50 no esta disponible; "
                    f"se usa {base_name} como base temporal provisional."
                )
                break

    if base_name is None:
        print("  AVISO: no hay ninguna serie euro area disponible para merge.")
        return pd.DataFrame(columns=["date"])

    df_monthly = data[base_name].copy()
    print(f"Base: {base_name} ({len(df_monthly)} meses)")

    for key, value in data.items():
        if key == base_name or value is None:
            continue
        df_monthly = pd.merge(df_monthly, value, on="date", how="left")
        print(f"  + {key:<28s} ({len(value)} meses)")

    print(f"\nDataFrame euro area: {len(df_monthly)} meses x {len(df_monthly.columns)} variables")
    return df_monthly


def calculate_euro_area_transformations(df):
    """
    Calcula transformaciones iniciales para el baseline euro area.

    No elimina las transformaciones US heredadas; esta funcion solo se usa
    cuando REGION='euro_area'.
    """
    print("\n" + "="*70)
    print("CALCULANDO TRANSFORMACIONES EURO AREA")
    print("="*70)

    if df.empty:
        print("  AVISO: DataFrame euro area vacio; no se calculan transformaciones.")
        return df

    print("\n  Logaritmos y rendimientos:")

    if "eurostoxx50" in df.columns:
        df["log_eurostoxx50"] = np.log(df["eurostoxx50"])
        df["ret_eurostoxx50"] = df["log_eurostoxx50"].diff()
        print("    OK log_eurostoxx50, ret_eurostoxx50")

    if "eurosystem_total_assets" in df.columns:
        clean_assets = df["eurosystem_total_assets"].where(df["eurosystem_total_assets"] > 0)
        df["log_eurosystem_total_assets"] = np.log(clean_assets)
        df["growth_eurosystem_total_assets"] = df["log_eurosystem_total_assets"].diff()
        print("    OK growth_eurosystem_total_assets")

    if "excess_liquidity" in df.columns:
        clean_excess_liquidity = df["excess_liquidity"].where(df["excess_liquidity"] > 0)
        df["log_excess_liquidity"] = np.log(clean_excess_liquidity)
        df["growth_excess_liquidity"] = df["log_excess_liquidity"].diff()
        print("    OK growth_excess_liquidity")

    print("\n  Diferencias simples:")

    if "vstoxx" in df.columns:
        df["delta_vstoxx"] = df["vstoxx"].diff()
        print("    OK delta_vstoxx")

    if "ciss" in df.columns:
        df["delta_ciss"] = df["ciss"].diff()
        print("    OK delta_ciss")

    if "deposit_facility_rate" in df.columns:
        df["delta_deposit_rate"] = df["deposit_facility_rate"].diff()
        print("    OK delta_deposit_rate")

    if "european_credit_spread" in df.columns:
        df["delta_european_credit_spread"] = df["european_credit_spread"].diff()
        print("    OK delta_european_credit_spread")

    print("\n  Variables derivadas:")

    if "euro_area_10y_yield" in df.columns and "euro_area_2y_yield" in df.columns:
        df["euro_area_slope_curve"] = df["euro_area_10y_yield"] - df["euro_area_2y_yield"]
        df["delta_euro_area_slope"] = df["euro_area_slope_curve"].diff()
        print("    OK euro_area_slope_curve, delta_euro_area_slope")

    if "emu_value" in df.columns and "emu_growth" in df.columns:
        df["emu_value_minus_growth"] = df["emu_value"] - df["emu_growth"]
        print("    OK emu_value_minus_growth")

    missing_summary = df.isnull().sum()
    missing_pct = (missing_summary / len(df) * 100).round(2)

    print("\n" + "="*70)
    print("RESUMEN DE VALORES FALTANTES - EURO AREA")
    print("="*70)
    print(f"\n{'Variable':<32} {'Missing':<10} {'%':<10}")
    print("-"*55)
    for var in df.columns:
        if var != "date":
            n_missing = missing_summary[var]
            pct_missing = missing_pct[var]
            if n_missing > 0:
                print(f"{var:<32} {n_missing:<10} {pct_missing:<10.2f}")

    return df


def summarize_raw_csv(filepath, value_col):
    """Resume cobertura del CSV bruto asociado a una variable."""
    if not filepath.exists():
        return {
            "raw_exists": False,
            "raw_observations": 0,
            "raw_start": "",
            "raw_end": "",
        }

    try:
        raw = pd.read_csv(filepath, parse_dates=["date"])
        if value_col not in raw.columns:
            return {
                "raw_exists": True,
                "raw_observations": len(raw),
                "raw_start": "",
                "raw_end": "",
            }
        valid = raw.dropna(subset=[value_col])
        return {
            "raw_exists": True,
            "raw_observations": int(len(valid)),
            "raw_start": valid["date"].min().strftime("%Y-%m-%d") if len(valid) else "",
            "raw_end": valid["date"].max().strftime("%Y-%m-%d") if len(valid) else "",
        }
    except Exception as exc:
        print(f"  AVISO: no se pudo auditar {filepath}: {exc}")
        return {
            "raw_exists": True,
            "raw_observations": "",
            "raw_start": "",
            "raw_end": "",
        }


def variable_coverage(df, variable):
    """Calcula cobertura mensual y missing values de una variable."""
    if df.empty or variable not in df.columns:
        return {
            "monthly_non_missing": 0,
            "missing_count": len(df) if not df.empty else 0,
            "missing_pct": 100.0 if not df.empty else "",
            "first_valid_date": "",
            "last_valid_date": "",
        }

    valid = df.dropna(subset=[variable])
    missing_count = int(df[variable].isna().sum())
    missing_pct = round(missing_count / len(df) * 100, 2) if len(df) else ""
    return {
        "monthly_non_missing": int(len(valid)),
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "first_valid_date": valid["date"].min().strftime("%Y-%m-%d") if len(valid) else "",
        "last_valid_date": valid["date"].max().strftime("%Y-%m-%d") if len(valid) else "",
    }


def build_euro_area_data_audit(df):
    """Construye tabla de auditoria de datos euro area."""
    rows = []

    transformation_notes = {
        "log_eurostoxx50": "log(eurostoxx50)",
        "ret_eurostoxx50": "diff(log_eurostoxx50)",
        "delta_vstoxx": "diff(vstoxx)",
        "delta_ciss": "diff(ciss)",
        "delta_deposit_rate": "diff(deposit_facility_rate)",
        "euro_area_slope_curve": "euro_area_10y_yield - euro_area_2y_yield",
        "delta_euro_area_slope": "diff(euro_area_slope_curve)",
        "log_eurosystem_total_assets": "log(eurosystem_total_assets) si estrictamente positivo",
        "growth_eurosystem_total_assets": "diff(log_eurosystem_total_assets)",
        "log_excess_liquidity": "log(excess_liquidity) si estrictamente positivo",
        "growth_excess_liquidity": "diff(log_excess_liquidity)",
        "emu_value_minus_growth": "emu_value - emu_growth si ambas series son comparables",
    }

    for variable, meta in EURO_AREA_MONTHLY_INPUTS.items():
        filepath = DATA_RAW / meta["filename"]
        raw_summary = summarize_raw_csv(filepath, variable)
        coverage = variable_coverage(df, variable)
        status = "confirmada" if meta.get("confirmed") else "pendiente/parcial"
        if meta.get("required") and variable not in df.columns:
            status = "faltante_requerida"

        rows.append({
            "variable": variable,
            "type": "raw",
            "source": meta.get("source", ""),
            "source_frequency": meta.get("source_frequency", ""),
            "aggregation": meta.get("method", ""),
            "required": bool(meta.get("required", False)),
            "source_status": status,
            "raw_file": meta["filename"],
            **raw_summary,
            **coverage,
            "transformation": "",
            "notes": meta.get("notes", ""),
        })

    for variable in transformation_notes:
        if variable not in df.columns:
            continue
        rows.append({
            "variable": variable,
            "type": "derived",
            "source": "pipeline",
            "source_frequency": "M",
            "aggregation": "",
            "required": False,
            "source_status": "calculada",
            "raw_file": "",
            "raw_exists": "",
            "raw_observations": "",
            "raw_start": "",
            "raw_end": "",
            **variable_coverage(df, variable),
            "transformation": transformation_notes[variable],
            "notes": "",
        })

    for variable, meta in EURO_AREA_PLACEHOLDER_SERIES.items():
        rows.append({
            "variable": variable,
            "type": "unresolved",
            "source": meta.get("source", ""),
            "source_frequency": "",
            "aggregation": "",
            "required": False,
            "source_status": "pendiente",
            "raw_file": "",
            "raw_exists": False,
            "raw_observations": 0,
            "raw_start": "",
            "raw_end": "",
            "monthly_non_missing": 0,
            "missing_count": "",
            "missing_pct": "",
            "first_valid_date": "",
            "last_valid_date": "",
            "transformation": "",
            "notes": meta.get("notes", ""),
        })

    return pd.DataFrame(rows)


def has_variable_coverage(df, variable):
    """Indica si una variable existe y tiene al menos una observacion valida."""
    return variable in df.columns and df[variable].notna().any()


def write_euro_area_audit_report(df, audit_df):
    """Escribe reporte markdown breve de auditoria de datos euro area."""
    report_path = PROJECT_ROOT / "EURO_AREA_DATA_AUDIT.md"

    if df.empty:
        sample_start = sample_end = ""
        monthly_obs = 0
    else:
        sample_start = df["date"].min().strftime("%Y-%m-%d")
        sample_end = df["date"].max().strftime("%Y-%m-%d")
        monthly_obs = len(df)

    missing_lines = []
    if not df.empty:
        for variable in df.columns:
            if variable == "date":
                continue
            missing_lines.append(
                f"- `{variable}`: {int(df[variable].isna().sum())} missing, "
                f"primer valido {variable_coverage(df, variable)['first_valid_date'] or 'n/a'}, "
                f"ultimo valido {variable_coverage(df, variable)['last_valid_date'] or 'n/a'}"
            )

    full_sample_core = [
        "eurostoxx50",
        "ciss",
        "deposit_facility_rate",
        "eurosystem_total_assets",
    ]
    full_sample_ready = all(has_variable_coverage(df, variable) for variable in full_sample_core)

    minimum_baseline_ready = all(
        has_variable_coverage(df, variable)
        for variable in [
            "growth_eurosystem_total_assets",
            "delta_ciss",
            "ret_eurostoxx50",
            "delta_deposit_rate",
        ]
    )
    excess_liquidity_ready = has_variable_coverage(df, "growth_excess_liquidity")
    growth_value_ready = has_variable_coverage(df, "emu_value_minus_growth")
    yield_curve_ready = has_variable_coverage(df, "euro_area_slope_curve")
    vstoxx_ready = has_variable_coverage(df, "delta_vstoxx")

    unresolved = audit_df[audit_df["source_status"].isin(["pendiente", "pendiente/parcial", "faltante_requerida"])]
    unresolved_lines = []
    seen_unresolved = set()
    for row in unresolved.itertuples():
        if row.variable in seen_unresolved or not getattr(row, "notes", ""):
            continue
        seen_unresolved.add(row.variable)
        unresolved_lines.append(f"- `{row.variable}`: {row.notes}")

    content = [
        "# Auditoria de datos euro area - Fase 2",
        "",
        "## Resumen de muestra",
        "",
        f"- Inicio de muestra mensual: `{sample_start or 'n/a'}`",
        f"- Fin de muestra mensual: `{sample_end or 'n/a'}`",
        f"- Observaciones mensuales: `{monthly_obs}`",
        f"- Soporte 2000-2026 con nucleo minimo usando activos totales: `{'si' if full_sample_ready else 'no'}`",
        f"- Modelo minimo listo con activos totales, CISS, EURO STOXX 50 y deposit facility rate: `{'si' if minimum_baseline_ready else 'no'}`",
        f"- Exceso de liquidez listo como baseline full-sample: `{'no' if not excess_liquidity_ready else 'solo muestra corta desde 2024'}`",
        f"- VSTOXX listo: `{'si' if vstoxx_ready else 'no'}`",
        f"- Growth/Value listo: `{'si' if growth_value_ready else 'no'}`",
        f"- Curva 2Y/10Y lista: `{'si, desde 2004' if yield_curve_ready else 'no'}`",
        "",
        "## Cobertura y missing values",
        "",
        *(missing_lines or ["- No hay datos mensuales disponibles."]),
        "",
        "## Decisiones de agregacion",
        "",
        "- Precios/indices de mercado: ultimo dato mensual cuando la fuente es diaria; EURO STOXX 50 usa ECB RTD mensual promedio por cobertura 2000-2026.",
        "- CISS y volatilidad: ultimo dato mensual por defecto; queda TODO probar promedio mensual como robustez.",
        "- Tipos de interes: ultimo dato mensual.",
        "- Balance/liquidez/tenencias: ultimo dato mensual.",
        "",
        "## Incidencias y pendientes",
        "",
        *(unresolved_lines or ["- No quedan fuentes pendientes registradas."]),
        "",
        f"Tabla CSV detallada: `results/tables/euro_area_data_audit.csv`",
    ]

    report_path.write_text("\n".join(content) + "\n", encoding="utf-8")
    print(f"  OK Reporte markdown guardado: {report_path}")

    return report_path


def save_euro_area_audit(df):
    """Guarda auditoria CSV y reporte markdown para el dataset euro area."""
    audit_df = build_euro_area_data_audit(df)
    output_audit = RESULTS_TABLES / "euro_area_data_audit.csv"
    audit_df.to_csv(output_audit, index=False, encoding="utf-8")
    print(f"  OK Auditoria CSV guardada: {output_audit}")
    write_euro_area_audit_report(df, audit_df)
    return audit_df


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
    if REGION == "euro_area":
        return load_euro_area_data()

    print("\n" + "="*70)
    print("CARGANDO Y RESAMPLING A MENSUAL")
    print("="*70)
    
    data = {}
    
    # -------------------------------------------------------------------------
    # 1. S&P 500 (diario → mensual, último valor)
    # -------------------------------------------------------------------------
    print("\n[1/12] S&P 500...")
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
    print("\n[2/12] VIX...")
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
    print("\n[3/12] Balance Fed...")
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
    print("\n[4/12] Fed Funds Rate...")
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
    print("\n[5/12] Treasury 2Y...")
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
    print("\n[6/12] Treasury 10Y...")
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
    print("\n[7/12] Spread BBB...")
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
    print("\n[8/12] GDP...")
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
    print("\n[9/12] Shiller CAPE...")
    shiller_path = DATA_EXTERNAL / "shiller_cape.csv"
    if shiller_path.exists():
        df_shiller = pd.read_csv(shiller_path, parse_dates=['date'])
        data['shiller'] = df_shiller[['date', 'price', 'dividend', 'earnings', 'cape']]
        print(f"  ✓ Shiller: {len(data['shiller'])} meses")
    else:
        
        print(f"  ✗ Archivo no encontrado: {shiller_path}")
        data['shiller'] = None
   
    # 10. Overnight Reverse Repo (diario → mensual, último valor)
    print("\n[10/12] Overnight Reverse Repo...")
    rrp_path = DATA_RAW / "fred_rrp_overnight.csv"
    if rrp_path.exists():
        data['rrp_overnight'] = load_and_resample(rrp_path, 'rrp_overnight', method='last')
        print(f"  ✓ Overnight Reverse Repo: {len(data['rrp_overnight'])} meses")
    else:
        print(f"  ✗ Archivo no encontrado: {rrp_path}")
        data['rrp_overnight'] = None
    
    # 11. Treasury General Account ( semanal → mensual, promedio))
    print("\n[11/12] Treasury General Account...")
    tga_path = DATA_RAW / "fred_tga.csv"
    if tga_path.exists():
        data['tga'] = load_and_resample(tga_path, 'tga', method='mean')
        print(f"  ✓ Treasury General Account: {len(data['tga'])} meses")
    else:
        print(f"  ✗ Archivo no encontrado: {tga_path}")
        data['tga'] = None


    # 12. Total Reserves (diario → mensual, último valor)
    print("\n[12/12] Total Reserves...")
    reserves_path = DATA_RAW / "fred_total_reserves.csv"
    if reserves_path.exists():
        data['total_reserves'] = load_and_resample(reserves_path, 'total_reserves', method='last')
        print(f"  ✓ Total Reserves: {len(data['total_reserves'])} meses")
    else:
        print(f"  ✗ Archivo no encontrado: {reserves_path}")
        data['total_reserves'] = None
    
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
    
    # RRP
    if data['rrp_overnight'] is not None:
        df_monthly = pd.merge(df_monthly, data['rrp_overnight'], on='date', how='left')
        print(f"  + rrp_overnight ({len(data['rrp_overnight'])} meses)")
    
    # TGA
    if data['tga'] is not None:
        df_monthly = pd.merge(df_monthly, data['tga'], on='date', how='left')
        print(f"  + tga ({len(data['tga'])} meses)")

    print(f"\nDataFrame final: {len(df_monthly)} meses × {len(df_monthly.columns)} variables")
    
    # Total Reserves
    if data['total_reserves'] is not None:
        df_monthly = pd.merge(df_monthly, data['total_reserves'], on='date', how='left')
        print(f"  + total_reserves ({len(data['total_reserves'])} meses)")

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

    # NET LIQUIDITY (Familia A)
    print("\n  Net Liquidity (Familia A):")

    if all(col in df.columns for col in ['fed_balance', 'rrp_overnight', 'tga']):
        # Reemplazar NaN en RRP y TGA con 0 (antes de 2021, no existían)
        df['rrp_overnight_filled'] = df['rrp_overnight'].fillna(0)
        df['tga_filled'] = df['tga'].fillna(0)

        #Net Liquidity = Balance - RRP - TGA
        df['net_liquidity'] = df['fed_balance'] - df['rrp_overnight_filled'] - df['tga_filled']
        print("    ✓ net_liquidity (Balance - RRP - TGA)")

        #Logaritmo
        df['log_net_liquidity'] = np.log(df['net_liquidity'])
        print("    ✓ log_net_liquidity")

        #Crecimiento mensual
        df['growth_net_liquidity'] = df['log_net_liquidity'].diff()
        print("    ✓ growth_net_liquidity (crecimiento mensual liquidez neta)")
    else:
         print("    ✗ Variables faltantes para calcular Net Liquidity")
    

    #RESERVAS TOTALES (FAMILIA B)
    print("\n  Total Reserves (Familia B):")

    if 'total_reserves' in df.columns:
        #loratimo
        df['log_total_reserves'] = np.log(df['total_reserves'])
        print("    ✓ log_total_reserves")

        #Crecimiento mensual
        df['growth_total_reserves'] = df['log_total_reserves'].diff()
        print("    ✓ growth_total_reserves (crecimiento mensual reservas totales)")   
    else:
        print("    ✗ Variable total_reserves no encontrado")


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
    print(f"Region activa: {REGION}")
    print(f"Periodo objetivo: {START_DATE} a {END_DATE}")
    print("="*70)
    
    start_time = datetime.now()
    
    # Paso 1: Cargar datos
    data = load_all_data()
    
    # Paso 2: Merge
    if REGION == "euro_area":
        df_monthly = merge_euro_area_series(data)
    else:
        df_monthly = merge_all_series(data)
    
    # Paso 3: Transformaciones
    if REGION == "euro_area":
        df_monthly = calculate_euro_area_transformations(df_monthly)
    else:
        df_monthly = calculate_transformations(df_monthly)

    if df_monthly.empty:
        print("\nAVISO: no hay datos mensuales disponibles. Pipeline finalizado sin guardar dataset.")
        return
    
    # Paso 4: Filtrar por rango de fechas
    print("\n" + "="*70)
    print("FILTRADO POR RANGO DE FECHAS")
    print("="*70)
    
    df_monthly = df_monthly[
        (df_monthly['date'] >= pd.to_datetime(START_DATE)) &
        (df_monthly['date'] <= pd.to_datetime(END_DATE))
    ]

    if df_monthly.empty:
        print("\nAVISO: no hay observaciones dentro del rango de fechas. No se guarda dataset.")
        return
    
    print(f"Observaciones en rango: {len(df_monthly)}")
    print(f"Periodo final: {df_monthly['date'].min().strftime('%Y-%m')} a "
          f"{df_monthly['date'].max().strftime('%Y-%m')}")
    
    # Paso 5: Guardar
    print("\n" + "="*70)
    print("GUARDANDO RESULTADOS")
    print("="*70)
    
    # CSV (human-readable)
    output_name = "monthly_data_euro_area" if REGION == "euro_area" else "monthly_data"
    output_csv = DATA_PROCESSED / f"{output_name}.csv"
    df_monthly.to_csv(output_csv, index=False)
    size_csv = output_csv.stat().st_size / 1024
    print(f"  OK CSV guardado: {output_csv} ({size_csv:.1f} KB)")
    
    # Pickle (Python-optimized)
    output_pkl = DATA_PROCESSED / f"{output_name}.pkl"
    df_monthly.to_pickle(output_pkl)
    size_pkl = output_pkl.stat().st_size / 1024
    print(f"  OK Pickle guardado: {output_pkl} ({size_pkl:.1f} KB)")

    if REGION == "euro_area":
        print("\n" + "="*70)
        print("AUDITORIA DE DATOS EURO AREA")
        print("="*70)
        save_euro_area_audit(df_monthly)
    
    # Resumen
    elapsed = (datetime.now() - start_time).total_seconds()
    print("\n" + "="*70)
    print("PIPELINE COMPLETADO")
    print("="*70)
    print(f"Tiempo total: {elapsed:.1f}s")
    print(f"Dataset final: {len(df_monthly)} meses x {len(df_monthly.columns)} variables")
    print("\nOK Datos listos para analisis\n")


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    main()
