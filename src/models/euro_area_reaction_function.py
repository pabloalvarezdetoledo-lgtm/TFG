"""
Scaffold de funcion de reaccion de liquidez para el area del euro.

Fase 3 solo prepara datos, validaciones y la formula candidata. La estimacion
final queda para Fase 4.
"""

import sys
from pathlib import Path

import pandas as pd


# Anadir src/ al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import DATA_PROCESSED


DATA_PATH = DATA_PROCESSED / "monthly_data_euro_area.csv"

DEPENDENT_VARIABLE = "growth_eurosystem_total_assets"
REACTION_FUNCTION_CONTROLS = [
    "ciss",
    "ret_eurostoxx50",
    "delta_deposit_rate",
]


def load_euro_area_data():
    """Carga el dataset mensual euro area para preparar la funcion de reaccion."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {DATA_PATH}. Ejecuta primero src/data_processing/create_monthly.py"
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def validate_reaction_function_inputs(df, min_obs=120):
    """Comprueba que las variables candidatas existen y tienen cobertura minima."""
    required = ["date", DEPENDENT_VARIABLE] + REACTION_FUNCTION_CONTROLS
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"AVISO: faltan variables para la funcion de reaccion: {missing}")
        return False

    valid = df[required].dropna()
    if len(valid) < min_obs:
        print(
            "AVISO: muestra util corta para funcion de reaccion "
            f"({len(valid)} obs; minimo sugerido {min_obs})"
        )
        return False

    print(f"OK muestra candidata funcion de reaccion: {len(valid)} observaciones")
    return True


def build_lagged_reaction_dataset(df, lags=6):
    """
    Construye un dataset rezagado para la especificacion futura.

    No estima el modelo. Solo deja preparada la matriz candidata:
    growth_eurosystem_total_assets_t sobre rezagos de liquidez, CISS,
    retornos del EURO STOXX 50 y cambios en el tipo de deposito.
    """
    required = ["date", DEPENDENT_VARIABLE] + REACTION_FUNCTION_CONTROLS
    available = [col for col in required if col in df.columns]
    work = df[available].copy()

    regressors = []
    for variable in [DEPENDENT_VARIABLE] + REACTION_FUNCTION_CONTROLS:
        if variable not in work.columns:
            continue
        for lag in range(1, lags + 1):
            lagged_name = f"{variable}_lag{lag}"
            work[lagged_name] = work[variable].shift(lag)
            regressors.append(lagged_name)

    model_columns = ["date", DEPENDENT_VARIABLE] + regressors
    model_df = work[model_columns].dropna().reset_index(drop=True)
    return model_df


def candidate_formula(lags=6):
    """Devuelve la formula conceptual recomendada para Fase 4."""
    terms = []
    for variable in [DEPENDENT_VARIABLE] + REACTION_FUNCTION_CONTROLS:
        lag_terms = [f"{variable}_lag{i}" for i in range(1, lags + 1)]
        terms.extend(lag_terms)
    return f"{DEPENDENT_VARIABLE} ~ " + " + ".join(terms)


def main():
    """Valida el scaffold de funcion de reaccion sin estimar modelos."""
    print("\n" + "=" * 70)
    print("SCAFFOLD FUNCION DE REACCION EURO AREA - FASE 3")
    print("=" * 70)

    df = load_euro_area_data()
    validate_reaction_function_inputs(df)
    model_df = build_lagged_reaction_dataset(df, lags=6)

    print(f"Formula candidata: {candidate_formula(lags=6)}")
    print(f"Observaciones tras rezagos: {len(model_df)}")
    print("TODO Fase 4: seleccionar rezagos, estimar OLS/HAC y reportar robustez.")


if __name__ == "__main__":
    main()
