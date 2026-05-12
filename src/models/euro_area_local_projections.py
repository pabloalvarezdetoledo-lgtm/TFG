"""
Scaffold de Local Projections para la especificacion euro area.

Fase 3 define variables, transformaciones y validaciones. No estima las LPs
finales ni genera IRFs.
"""

import sys
from pathlib import Path

import pandas as pd


# Anadir src/ al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import DATA_PROCESSED


DATA_PATH = DATA_PROCESSED / "monthly_data_euro_area.csv"

SHOCK_VARIABLE = "growth_eurosystem_total_assets"
RESPONSE_VARIABLES = [
    "ret_eurostoxx50",
    "delta_ciss",
    "ciss",
    "delta_deposit_rate",
]
CONTROL_VARIABLES = [
    "ciss",
    "ret_eurostoxx50",
    "delta_deposit_rate",
]


def load_euro_area_data():
    """Carga el dataset mensual euro area para preparar LPs."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {DATA_PATH}. Ejecuta primero src/data_processing/create_monthly.py"
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def validate_local_projection_inputs(df, min_obs=120):
    """Comprueba cobertura minima para respuestas, shock y controles."""
    required = ["date", SHOCK_VARIABLE] + RESPONSE_VARIABLES + CONTROL_VARIABLES
    required = list(dict.fromkeys(required))
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"AVISO: faltan variables para Local Projections euro area: {missing}")
        return False

    valid = df[required].dropna()
    if len(valid) < min_obs:
        print(
            "AVISO: muestra util corta para Local Projections "
            f"({len(valid)} obs; minimo sugerido {min_obs})"
        )
        return False

    print(f"OK muestra candidata LP: {len(valid)} observaciones")
    return True


def build_lp_design(df, response_var, horizon=0, lags=6):
    """
    Construye el dataset candidato para una LP sin estimarla.

    Variable dependiente provisional:
        y_{t+h} - y_{t-1}

    Esta eleccion se revisara en Fase 4 segun la variable respuesta y su
    interpretacion economica.
    """
    if response_var not in RESPONSE_VARIABLES:
        raise ValueError(f"Respuesta no configurada: {response_var}")

    required = ["date", SHOCK_VARIABLE, response_var] + CONTROL_VARIABLES
    required = list(dict.fromkeys(required))
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para LP: {missing}")

    work = df[required].copy()
    dep_var = f"{response_var}_h{horizon}_change"
    work[dep_var] = work[response_var].shift(-horizon) - work[response_var].shift(1)

    regressors = [SHOCK_VARIABLE]
    for variable in [SHOCK_VARIABLE] + CONTROL_VARIABLES:
        for lag in range(1, lags + 1):
            lagged_name = f"{variable}_lag{lag}"
            work[lagged_name] = work[variable].shift(lag)
            regressors.append(lagged_name)

    design_columns = ["date", dep_var] + regressors
    design_df = work[design_columns].dropna().reset_index(drop=True)
    return design_df


def candidate_equation(response_var, horizon):
    """Devuelve la ecuacion conceptual LP recomendada."""
    return (
        f"{response_var}_{{t+{horizon}}} - {response_var}_{{t-1}} = "
        "alpha_h + beta_h shock_liquidity_t + controls_t + error_{t+h}"
    )


def main():
    """Valida el scaffold LP sin estimar modelos finales."""
    print("\n" + "=" * 70)
    print("SCAFFOLD LOCAL PROJECTIONS EURO AREA - FASE 3")
    print("=" * 70)

    df = load_euro_area_data()
    validate_local_projection_inputs(df)

    for response in RESPONSE_VARIABLES:
        design_df = build_lp_design(df, response_var=response, horizon=0, lags=6)
        print(f"{response}: {len(design_df)} observaciones candidatas en h=0")

    print(f"Ecuacion candidata: {candidate_equation('ret_eurostoxx50', 0)}")
    print("TODO Fase 4: definir shock exogeno, horizontes, errores HAC y controles.")


if __name__ == "__main__":
    main()
