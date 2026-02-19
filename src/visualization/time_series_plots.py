"""
Visualización de series temporales
Crea gráficos profesionales de las series en niveles y transformadas

Autor: Pablo Álvarez de Toledo Rodríguez
Fecha: Enero 2026
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Añadir src/ al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import (
    DATA_PROCESSED,
    RESULTS_FIGURES,
    EVENTS,
    PLOT_STYLE,
    FIGURE_DPI,
    FIGURE_FORMAT
)


# =============================================================================
# CONFIGURACIÓN DE MATPLOTLIB
# =============================================================================

# Establecer estilo
plt.style.use(PLOT_STYLE)

# Configuración global de fuentes y tamaños
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Configuración de formato de fechas
plt.rcParams['date.autoformatter.year'] = '%Y'
plt.rcParams['date.autoformatter.month'] = '%Y-%m'


# =============================================================================
# FUNCIÓN AUXILIAR: MARCAR EVENTOS
# =============================================================================

def add_event_markers(ax, events_dict, y_position='top', alpha=0.3):
    """
    Añade líneas verticales y etiquetas para eventos importantes
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Eje donde añadir los marcadores
    events_dict : dict
        Diccionario con eventos {'nombre': 'YYYY-MM-DD'}
    y_position : str
        Posición vertical de etiquetas: 'top', 'bottom', 'middle'
    alpha : float
        Transparencia de las líneas verticales (0-1)
    
    Notes
    -----
    Esta función permite identificar visualmente periodos clave:
    - QE1, QE2, QE3: Expansión del balance
    - Taper tantrum: Anuncio de reducción de compras
    - COVID: Shock exógeno masivo
    - Rate hikes: Normalización monetaria
    """
    # Colores para diferentes tipos de eventos
    event_colors = {
        'QE': '#2ca02c',      # Verde para expansión
        'Taper': '#d62728',   # Rojo para contracción
        'COVID': '#ff7f0e',   # Naranja para crisis
        'Rate': '#9467bd',    # Púrpura para política de tipos
        'SVB': '#8c564b'      # Marrón para crisis bancaria
    }
    
    # Determinar color según tipo de evento
    def get_event_color(event_name):
        if 'QE' in event_name or 'Operation' in event_name:
            return event_colors['QE']
        elif 'Taper' in event_name:
            return event_colors['Taper']
        elif 'COVID' in event_name:
            return event_colors['COVID']
        elif 'rate' in event_name or 'Rate' in event_name:
            return event_colors['Rate']
        elif 'SVB' in event_name:
            return event_colors['SVB']
        else:
            return '#7f7f7f'  # Gris por defecto
    
    # Añadir líneas verticales para cada evento
    for event_name, event_date in events_dict.items():
        event_datetime = pd.to_datetime(event_date)
        color = get_event_color(event_name)
        
        # Línea vertical
        ax.axvline(x=event_datetime, color=color, linestyle='--', 
                   linewidth=1, alpha=alpha, zorder=1)
        
        # Etiqueta (solo para eventos principales para no saturar)
        main_events = ['QE1_announcement', 'QE3_announcement', 
                       'Taper_tantrum', 'COVID_QE_unlimited', 
                       'First_rate_hike']
        
        if event_name in main_events:
            # Determinar posición Y de la etiqueta
            ylim = ax.get_ylim()
            if y_position == 'top':
                y_text = ylim[1] * 0.95
                va = 'top'
            elif y_position == 'bottom':
                y_text = ylim[0] + (ylim[1] - ylim[0]) * 0.05
                va = 'bottom'
            else:  # middle
                y_text = ylim[0] + (ylim[1] - ylim[0]) * 0.5
                va = 'center'
            
            # Etiqueta simplificada
            label = event_name.replace('_announcement', '').replace('_', ' ')
            
            ax.text(event_datetime, y_text, label,
                   rotation=90, fontsize=8, alpha=0.7,
                   verticalalignment=va, horizontalalignment='right',
                   color=color)


# =============================================================================
# GRÁFICO 1: SERIES EN NIVELES
# =============================================================================

def plot_series_levels(df):
    """
    Crea gráfico de 4 paneles con series principales en niveles
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos mensuales
    
    Returns
    -------
    matplotlib.figure.Figure
        Objeto figura de matplotlib
        
    Notes
    -----
    Panel superior izquierdo: S&P 500 (log scale)
    Panel superior derecho: Balance de la Fed (log scale)
    Panel inferior izquierdo: VIX (volatilidad)
    Panel inferior derecho: Spread BBB (riesgo de crédito)
    
    Justificación de log scale para S&P y Balance:
    - Crecimiento exponencial en el tiempo
    - Log scale muestra tasas de crecimiento constantes como líneas rectas
    - Más fácil comparar periodos de alta y baja volatilidad
    """
    print("\n[1/3] Creando gráfico de series en niveles...")
    
    # Crear figura con 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Series Temporales en Niveles (2000-2025)', 
                 fontsize=16, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Panel 1: S&P 500 (escala logarítmica)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(df['date'], df['sp500'], linewidth=1.5, color='#1f77b4', label='S&P 500')
    ax1.set_ylabel('S&P 500 Index', fontweight='bold')
    ax1.set_yscale('log')  # Escala logarítmica
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Formatear eje Y con separadores de miles
    from matplotlib.ticker import FuncFormatter
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Añadir eventos
    add_event_markers(ax1, EVENTS, y_position='top', alpha=0.2)
    
    # -------------------------------------------------------------------------
    # Panel 2: Balance de la Fed (escala logarítmica)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.plot(df['date'], df['fed_balance'], linewidth=1.5, 
             color='#2ca02c', label='Balance Fed')
    ax2.set_ylabel('Fed Balance Sheet (millones USD)', fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Formatear eje Y con separadores de miles
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Añadir eventos
    add_event_markers(ax2, EVENTS, y_position='top', alpha=0.2)
    
    # -------------------------------------------------------------------------
    # Panel 3: VIX (volatilidad implícita)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.plot(df['date'], df['vix'], linewidth=1.5, color='#ff7f0e', label='VIX')
    ax3.set_ylabel('VIX Index', fontweight='bold')
    ax3.set_xlabel('Fecha', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    
    # Línea horizontal en VIX = 20 (umbral de "alta volatilidad")
    ax3.axhline(y=20, color='red', linestyle=':', linewidth=1, 
                alpha=0.5, label='Umbral 20')
    
    # Añadir eventos
    add_event_markers(ax3, EVENTS, y_position='top', alpha=0.2)
    
    # -------------------------------------------------------------------------
    # Panel 4: Spread de crédito BBB
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.plot(df['date'], df['spread_bbb'], linewidth=1.5, 
             color='#d62728', label='Spread BBB')
    ax4.set_ylabel('BBB OAS (basis points)', fontweight='bold')
    ax4.set_xlabel('Fecha', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    
    # Añadir eventos
    add_event_markers(ax4, EVENTS, y_position='top', alpha=0.2)
    
    # -------------------------------------------------------------------------
    # Formatear ejes X (fechas) para todos los paneles
    # -------------------------------------------------------------------------
    for ax in axes.flat:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Ticks cada 5 años
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))  # Ticks menores cada año
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)
    
    # Ajustar espaciado
    plt.tight_layout()
    
    # Guardar
    output_path = RESULTS_FIGURES / f'fig_series_levels.{FIGURE_FORMAT}'
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  ✓ Guardado: {output_path}")
    
    return fig


# =============================================================================
# GRÁFICO 2: RENDIMIENTOS Y CRECIMIENTOS (SERIES TRANSFORMADAS)
# =============================================================================

def plot_series_returns(df):
    """
    Crea gráfico de 3 paneles con rendimientos y crecimientos
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos mensuales
    
    Returns
    -------
    matplotlib.figure.Figure
        Objeto figura de matplotlib
        
    Notes
    -----
    Panel superior: Rendimiento S&P 500 (Δlog)
    Panel medio: Crecimiento Balance Fed (Δlog)
    Panel inferior: Cambio en VIX (diferencia simple)
    
    Estas son las variables que usaremos en XGBoost y análisis de corto plazo.
    Los cambios extremos (outliers) son informativos, no errores:
    - Marzo 2020 (COVID): Caída S&P 500 ~35%, VIX spike ~500%
    - Crisis 2008: Múltiples shocks
    """
    print("\n[2/3] Creando gráfico de rendimientos y crecimientos...")
    
    # Crear figura con 3 subplots verticales
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Rendimientos y Crecimientos Mensuales (2000-2025)', 
                 fontsize=16, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Panel 1: Rendimiento S&P 500
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.bar(df['date'], df['ret_sp500'] * 100, width=20, 
            color='#1f77b4', alpha=0.7, label='Rendimiento S&P 500')
    ax1.set_ylabel('Rendimiento mensual (%)', fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(loc='upper left')
    
    # Añadir eventos
    add_event_markers(ax1, EVENTS, y_position='top', alpha=0.15)
    
    # -------------------------------------------------------------------------
    # Panel 2: Crecimiento Balance Fed
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    
    # Colorear según signo (verde = expansión, rojo = contracción)
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in df['growth_balance']]
    
    ax2.bar(df['date'], df['growth_balance'] * 100, width=20, 
            color=colors, alpha=0.7, label='Crecimiento Balance Fed')
    ax2.set_ylabel('Crecimiento mensual (%)', fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper left')
    
    # Añadir eventos
    add_event_markers(ax2, EVENTS, y_position='top', alpha=0.15)
    
    # -------------------------------------------------------------------------
    # Panel 3: Cambio en VIX
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    ax3.bar(df['date'], df['delta_vix'], width=20, 
            color='#ff7f0e', alpha=0.7, label='Cambio en VIX')
    ax3.set_ylabel('Δ VIX (puntos)', fontweight='bold')
    ax3.set_xlabel('Fecha', fontweight='bold')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(loc='upper left')
    
    # Añadir eventos
    add_event_markers(ax3, EVENTS, y_position='top', alpha=0.15)
    
    # -------------------------------------------------------------------------
    # Formatear ejes X (fechas)
    # -------------------------------------------------------------------------
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='x', rotation=45)
    
    # Ajustar espaciado
    plt.tight_layout()
    
    # Guardar
    output_path = RESULTS_FIGURES / f'fig_series_returns.{FIGURE_FORMAT}'
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  ✓ Guardado: {output_path}")
    
    return fig


# =============================================================================
# GRÁFICO 3: BALANCE FED vs S&P 500 (EJES DUALES)
# =============================================================================

def plot_balance_vs_sp500(df):
    """
    Crea gráfico de ejes duales: Balance Fed vs S&P 500
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos mensuales
    
    Returns
    -------
    matplotlib.figure.Figure
        Objeto figura de matplotlib
        
    Notes
    -----
    Este gráfico es el "money shot" del TFG.
    Muestra visualmente la correlación entre:
    - Expansión del balance de la Fed (eje izquierdo, verde)
    - Valoración del S&P 500 (eje derecho, azul)
    
    La correlación visual es impactante:
    - 2008-2014: QE → Balance sube → S&P sube
    - 2017-2019: Balance plano → S&P sube moderadamente
    - 2020-2021: QE masivo → S&P rally vertical
    - 2022-2023: QT (balance baja) → S&P corrección
    
    Este gráfico irá en la introducción del TFG para motivar la pregunta
    de investigación: ¿Es esta correlación causal?
    """
    print("\n[3/3] Creando gráfico Balance Fed vs S&P 500...")
    
    # Crear figura
    fig, ax1 = plt.subplots(figsize=(14, 7))
    fig.suptitle('Liquidez de la Fed y Valoración del S&P 500 (2000-2025)', 
                 fontsize=16, fontweight='bold')
    
    # -------------------------------------------------------------------------
    # Eje izquierdo: Balance de la Fed
    # -------------------------------------------------------------------------
    color_balance = '#2ca02c'
    ax1.set_xlabel('Fecha', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Balance Fed (millones USD)', fontweight='bold', 
                   fontsize=12, color=color_balance)
    
    # Plot del balance con área sombreada
    ax1.plot(df['date'], df['fed_balance'], linewidth=2.5, 
             color=color_balance, label='Balance Fed', zorder=3)
    ax1.fill_between(df['date'], df['fed_balance'], alpha=0.2, 
                     color=color_balance, zorder=1)
    
    ax1.tick_params(axis='y', labelcolor=color_balance)
    ax1.set_yscale('log')  # Escala logarítmica
    ax1.grid(True, alpha=0.2, zorder=0)
    
    # Formatear eje Y1
    from matplotlib.ticker import FuncFormatter
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1000):,}k'))
    
    # -------------------------------------------------------------------------
    # Eje derecho: S&P 500
    # -------------------------------------------------------------------------
    ax2 = ax1.twinx()  # Crear segundo eje Y que comparte eje X
    
    color_sp500 = '#1f77b4'
    ax2.set_ylabel('S&P 500 Index', fontweight='bold', 
                   fontsize=12, color=color_sp500)
    
    ax2.plot(df['date'], df['sp500'], linewidth=2.5, 
             color=color_sp500, label='S&P 500', zorder=3, linestyle='--')
    
    ax2.tick_params(axis='y', labelcolor=color_sp500)
    ax2.set_yscale('log')  # Escala logarítmica
    
    # Formatear eje Y2
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # -------------------------------------------------------------------------
    # Añadir eventos (solo los más importantes para no saturar)
    # -------------------------------------------------------------------------
    key_events = {
        'QE1_announcement': EVENTS['QE1_announcement'],
        'QE3_announcement': EVENTS['QE3_announcement'],
        'Taper_tantrum': EVENTS['Taper_tantrum'],
        'COVID_QE_unlimited': EVENTS['COVID_QE_unlimited'],
        'First_rate_hike': EVENTS['First_rate_hike']
    }
    
    add_event_markers(ax1, key_events, y_position='top', alpha=0.25)
    
    # -------------------------------------------------------------------------
    # Formatear eje X (fechas)
    # -------------------------------------------------------------------------
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.tick_params(axis='x', rotation=45)
    
    # -------------------------------------------------------------------------
    # Leyenda combinada
    # -------------------------------------------------------------------------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper left', fontsize=11, framealpha=0.9)
    
    # Ajustar layout
    fig.tight_layout()
    
    # Guardar
    output_path = RESULTS_FIGURES / f'fig_balance_vs_sp500.{FIGURE_FORMAT}'
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"  ✓ Guardado: {output_path}")
    
    return fig


# =============================================================================
# FUNCIÓN PRINCIPAL
# =============================================================================

def main():
    """
    Ejecuta pipeline completo de visualización
    """
    print("\n" + "="*70)
    print(" PIPELINE DE VISUALIZACIÓN DE SERIES TEMPORALES")
    print("="*70)
    
    # Cargar datos
    print("\nCargando datos...")
    data_path = DATA_PROCESSED / "monthly_data.csv"
    df = pd.read_csv(data_path, parse_dates=['date'])
    
    print(f"  ✓ Datos cargados: {len(df)} observaciones")
    print(f"  Periodo: {df['date'].min().strftime('%Y-%m')} a "
          f"{df['date'].max().strftime('%Y-%m')}")
    
    # Crear gráficos
    print("\n" + "="*70)
    print("GENERANDO GRÁFICOS")
    print("="*70)
    
    fig1 = plot_series_levels(df)
    fig2 = plot_series_returns(df)
    fig3 = plot_balance_vs_sp500(df)
    
    # Resumen
    print("\n" + "="*70)
    print("VISUALIZACIÓN COMPLETADA")
    print("="*70)
    print(f"Gráficos guardados en: {RESULTS_FIGURES}")
    print(f"Formato: {FIGURE_FORMAT}")
    print(f"Resolución: {FIGURE_DPI} DPI")
    print("\nArchivos creados:")
    print(f"  - fig_series_levels.{FIGURE_FORMAT}")
    print(f"  - fig_series_returns.{FIGURE_FORMAT}")
    print(f"  - fig_balance_vs_sp500.{FIGURE_FORMAT}")
    print("\n✓ Pipeline completado\n")
    
    # Cerrar figuras para liberar memoria
    plt.close('all')


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    main()