"""
Graficos descriptivos para el dataset mensual euro area.

Este script crea figuras de diagnostico visual para la Fase 3. No estima
modelos ni reemplaza los graficos US heredados.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


# Anadir src/ al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import DATA_PROCESSED, FIGURE_DPI, PLOT_STYLE, RESULTS_FIGURES


DATA_PATH = DATA_PROCESSED / "monthly_data_euro_area.csv"


plt.style.use(PLOT_STYLE)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 14


def load_euro_area_data():
    """Carga el dataset mensual euro area usado en Fase 3."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {DATA_PATH}. Ejecuta primero src/data_processing/create_monthly.py"
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"OK datos cargados: {len(df)} observaciones")
    return df


def warn_missing_columns(df, columns, figure_name):
    """Devuelve columnas faltantes y muestra un aviso claro."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        print(f"AVISO: no se crea {figure_name}; faltan columnas: {missing}")
    return missing


def format_date_axis(ax):
    """Aplica formato uniforme al eje temporal."""
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.25)


def save_figure(fig, filename):
    """Guarda una figura en results/figures con nombre euro_area_*."""
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_FIGURES / filename
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"OK figura guardada: {output_path}")
    return output_path


def plot_eurostoxx50_liquidity(df):
    """Grafica EURO STOXX 50 y activos totales del Eurosistema."""
    filename = "euro_area_eurostoxx50_liquidity.png"
    required = ["date", "eurostoxx50", "eurosystem_total_assets"]
    if warn_missing_columns(df, required, filename):
        return None

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(
        df["date"],
        df["eurosystem_total_assets"],
        color="#0b6e4f",
        linewidth=2.0,
        label="Activos totales Eurosistema",
    )
    ax2.plot(
        df["date"],
        df["eurostoxx50"],
        color="#2f5f98",
        linewidth=1.8,
        label="EURO STOXX 50",
    )

    ax1.set_title("EURO STOXX 50 y liquidez del Eurosistema")
    ax1.set_ylabel("Activos totales Eurosistema")
    ax2.set_ylabel("EURO STOXX 50")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    format_date_axis(ax1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    return save_figure(fig, filename)


def plot_ciss_liquidity(df):
    """Grafica CISS y crecimiento de activos totales del Eurosistema."""
    filename = "euro_area_ciss_liquidity.png"
    required = ["date", "ciss", "growth_eurosystem_total_assets"]
    if warn_missing_columns(df, required, filename):
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(df["date"], df["ciss"], color="#8a3ffc", linewidth=1.7, label="CISS")
    axes[0].set_title("Estres sistemico y crecimiento de liquidez")
    axes[0].set_ylabel("CISS")
    axes[0].legend(loc="upper left")

    colors = ["#0b6e4f" if value >= 0 else "#c43b3b" for value in df["growth_eurosystem_total_assets"].fillna(0)]
    axes[1].bar(
        df["date"],
        df["growth_eurosystem_total_assets"] * 100,
        width=22,
        color=colors,
        alpha=0.75,
        label="Crecimiento activos Eurosistema",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Crecimiento mensual (%)")
    axes[1].set_xlabel("Fecha")
    axes[1].legend(loc="upper left")

    for ax in axes:
        format_date_axis(ax)

    fig.tight_layout()
    return save_figure(fig, filename)


def plot_policy_rate_liquidity(df):
    """Grafica tipo de deposito del ECB y crecimiento de liquidez."""
    filename = "euro_area_policy_rate_liquidity.png"
    required = ["date", "deposit_facility_rate", "growth_eurosystem_total_assets"]
    if warn_missing_columns(df, required, filename):
        return None

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(
        df["date"],
        df["deposit_facility_rate"],
        color="#6c4f3d",
        linewidth=1.8,
        label="Deposit facility rate",
    )
    ax2.bar(
        df["date"],
        df["growth_eurosystem_total_assets"] * 100,
        width=22,
        color="#3c8d7b",
        alpha=0.45,
        label="Crecimiento activos Eurosistema",
    )

    ax1.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    ax2.axhline(0, color="#3c8d7b", linewidth=0.8, alpha=0.5)
    ax1.set_title("Politica de tipos ECB y liquidez del Eurosistema")
    ax1.set_ylabel("Deposit facility rate (%)")
    ax2.set_ylabel("Crecimiento mensual activos (%)")
    format_date_axis(ax1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    return save_figure(fig, filename)


def plot_returns_ciss(df):
    """Grafica retornos del EURO STOXX 50 y CISS."""
    filename = "euro_area_returns_ciss.png"
    required = ["date", "ret_eurostoxx50", "ciss"]
    if warn_missing_columns(df, required, filename):
        return None

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    colors = ["#2f5f98" if value >= 0 else "#c43b3b" for value in df["ret_eurostoxx50"].fillna(0)]
    axes[0].bar(
        df["date"],
        df["ret_eurostoxx50"] * 100,
        width=22,
        color=colors,
        alpha=0.8,
        label="Retorno EURO STOXX 50",
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Retornos de mercado y estres sistemico")
    axes[0].set_ylabel("Retorno mensual (%)")
    axes[0].legend(loc="upper left")

    axes[1].plot(df["date"], df["ciss"], color="#8a3ffc", linewidth=1.7, label="CISS")
    axes[1].set_ylabel("CISS")
    axes[1].set_xlabel("Fecha")
    axes[1].legend(loc="upper left")

    for ax in axes:
        format_date_axis(ax)

    fig.tight_layout()
    return save_figure(fig, filename)


def main():
    """Ejecuta los graficos descriptivos euro area de Fase 3."""
    print("\n" + "=" * 70)
    print("GRAFICOS EURO AREA - FASE 3")
    print("=" * 70)

    df = load_euro_area_data()
    created = [
        plot_eurostoxx50_liquidity(df),
        plot_ciss_liquidity(df),
        plot_policy_rate_liquidity(df),
        plot_returns_ciss(df),
    ]

    created = [path for path in created if path is not None]
    print(f"\nOK graficos creados: {len(created)}")


if __name__ == "__main__":
    main()
