# Arquitectura de la Liquidez: Expansión Monetaria, Liquidez Neta y Valoración de Activos

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Trabajo de Fin de Grado en Economía y Negocios Internacionales, Universidad Antonio de Nebrija  
**Autor:** Pablo Álvarez de Toledo Rodríguez  
**Tutora:** Raquel García-Donás Guerrero  
**Periodo:** Enero-Junio 2026

---

## Tabla de contenidos

- [Resumen del proyecto](#resumen-del-proyecto)
- [Pregunta de investigación](#pregunta-de-investigación)
- [Hipótesis de trabajo](#hipótesis-de-trabajo)
- [Estructura actual del repositorio](#estructura-actual-del-repositorio)
- [Instalación](#instalación)
- [Pipeline de trabajo](#pipeline-de-trabajo)
- [Marco teórico](#marco-teórico)
- [Metodología empírica actual](#metodología-empírica-actual)
- [Datos](#datos)
- [Resultados esperados y estado del proyecto](#resultados-esperados-y-estado-del-proyecto)
- [Líneas futuras](#líneas-futuras)
- [Licencia](#licencia)
- [Contacto](#contacto)

---

## Resumen del proyecto

Este repositorio contiene el desarrollo empírico de mi Trabajo de Fin de Grado, centrado en el análisis de la relación entre política monetaria expansiva, liquidez del sistema financiero y valoración de los mercados bursátiles en Estados Unidos.

La idea central del proyecto es que la variable monetaria relevante no es simplemente el tamaño bruto del balance de la Reserva Federal, sino la **liquidez neta efectivamente disponible para el sistema financiero privado**, definida como:

\[
NetLiq_t = WALCL_t - RRP_t - TGA_t
\]

donde:

- `WALCL` representa el tamaño total del balance de la Reserva Federal,
- `RRP` recoge la absorción de liquidez vía Overnight Reverse Repo Facility,
- `TGA` es la Treasury General Account del Tesoro estadounidense.

Desde esta perspectiva, el proyecto intenta ir más allá de la correlación visual entre balance de la Fed y S&P 500, y construir un marco econométrico más riguroso que permita distinguir entre relaciones de largo plazo, efectos dinámicos de corto plazo y distintas familias de proxies de liquidez. La línea principal actualmente implementada trabaja con **Net Liquidity** como proxy central, y con **reservas totales** como familia alternativa de contraste. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

---

## Pregunta de investigación

La pregunta principal del trabajo es la siguiente:

**¿Influye la liquidez neta generada por la Reserva Federal en la valoración del mercado bursátil estadounidense y en la compresión de primas de riesgo financieras?**

De forma más concreta, el proyecto estudia si las expansiones monetarias alteran las condiciones financieras y sostienen valoraciones de renta variable a través de canales de liquidez, compresión de spreads y reducción de volatilidad, y si estos efectos aparecen en relaciones de largo plazo o, más bien, en dinámicas de corto plazo. :contentReference[oaicite:6]{index=6}

---

## Hipótesis de trabajo

El diseño teórico del TFG se organiza en torno a cinco hipótesis principales:

**H1. Función de reacción endógena.**  
La Reserva Federal expande su balance en respuesta a deterioros de las condiciones financieras, como aumentos del VIX, ampliaciones de spreads o caídas bursátiles.

**H2. Cointegración de largo plazo.**  
Existe una relación de equilibrio de largo plazo entre valoración bursátil, fundamentales y liquidez, especialmente cuando esta última se mide como Net Liquidity en lugar de balance bruto.

**H3. Canal de compresión de primas.**  
Los shocks positivos de liquidez reducen spreads de crédito y volatilidad, y favorecen los rendimientos bursátiles.

**H4. Dependencia de régimen.**  
El impacto de la liquidez es más intenso en episodios de estrés financiero o alta volatilidad.

**H5. Relevancia predictiva fuera de muestra.**  
Las innovaciones de liquidez contienen información útil para anticipar dinámica bursátil y condiciones financieras.

Aunque el marco conceptual del anteproyecto contempla también extensiones con Markov Switching, SVAR y XGBoost, la implementación actual del repositorio está concentrada en la construcción del dataset, tests de integración, VECM y VAR. :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

---

## Estructura actual del repositorio

```text
tfg-liquidez-arquitectura/
│
├── README.md
├── .gitignore
├── requirements.txt
├── LICENSE
│
├── data/
│   ├── raw/                  # Datos originales descargados
│   ├── processed/            # monthly_data.csv y ficheros procesados
│   └── external/             # Dataset de Shiller y otras fuentes externas
│
├── src/
│   ├── data_processing/
│   │   ├── download_data.py
│   │   └── create_monthly.py
│   │
│   ├── diagnostics/
│   │   └── unit_root_tests.py
│   │
│   ├── models/
│   │   ├── VECM_Net_Liquidity.py
│   │   ├── VECM_2_NetLiq.py
│   │   ├── VECM_Reservas_familiaB.py
│   │   └── VAR_en_Diferencias.py
│   │
│   ├── visualization/
│   │   └── time_series_plots.py
│   │
│   └── utils/
│       └── config.py
│
└── results/
    ├── tables/
    ├── figures/
    └── models/
