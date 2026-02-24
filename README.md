# Arquitectura de la Liquidez: Políticas Monetarias Expansivas y Valoración de Activos

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Trabajo de Fin de Grado en Economía - Universidad Antonio Nebrija  
**Autor:** Pablo Álvarez de Toledo Rodríguez  
**Director:** Raquel Garcia Donas Guerrero   
**Fecha:** Enero-Junio 2026

---

## 📋 Tabla de Contenidos

- [Resumen Ejecutivo](#resumen-ejecutivo)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso Rápido](#uso-rápido)
- [Marco Teórico](#marco-teórico)
- [Metodología Empírica](#metodología-empírica)
- [Datos](#datos)
- [Resultados Principales](#resultados-principales)
- [Replicación](#replicación)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)
- [Contacto](#contacto)

---

## 🎯 Resumen Ejecutivo

Este proyecto investiga la relación entre las **expansiones de liquidez** implementadas por la Reserva Federal y la **valoración de activos de renta variable** (S&P 500) durante el periodo 2000-2025.

### Pregunta de Investigación

¿Alteran las políticas monetarias expansivas (Quantitative Easing) las señales intertemporales que coordinan ahorro e inversión, generando valoraciones de activos inconsistentes con sus fundamentos de largo plazo?

### Hipótesis Principales

1. **H1 (Función de reacción endógena):** El balance de la Fed responde endógenamente a condiciones financieras adversas (↑VIX, ↑Spread, ↓S&P 500)

2. **H2 (Cointegración estructural):** Existe una relación de largo plazo entre liquidez (`log(Balance)`), fundamentales (`log(Earnings)`) y valoración (`log(S&P 500)`)

3. **H3 (Canal de compresión de primas):** Shocks de liquidez reducen primas de riesgo (VIX, spreads) y elevan valoraciones (IRF positivo)

4. **H4 (Dependencia de régimen):** El impacto de la liquidez es mayor en regímenes de alta volatilidad (crisis)

5. **H5 (Relevancia predictiva):** La liquidez aporta contenido informativo significativo fuera de muestra

### Metodología

- **VECM (Vector Error Correction Model):** Relación de largo plazo y dinámica de ajuste
- **HMM (Hidden Markov Model):** Identificación de regímenes de mercado (bull/bear)
- **XGBoost + SHAP:** Modelado no lineal de corto plazo e interpretabilidad
- **Proyecciones Locales (Jordà, 2005):** Efectos heterogéneos por régimen
- **Event Study:** Análisis de anuncios de QE

---

## 📁 Estructura del Proyecto
```
tfg-liquidez-arquitectura/
│
├── README.md                          # Este archivo
├── .gitignore                         # Archivos ignorados por Git
├── requirements.txt                   # Dependencias Python
├── LICENSE                            # Licencia MIT
│
├── data/                              # Datos (no versionados en Git)
│   ├── raw/                           # Datos originales (FRED, Yahoo, Shiller)
│   ├── processed/                     # monthly_data.csv (agregado mensual)
│   └── external/                      # Shiller CAPE dataset
│
├── src/                               # Código fuente
│   ├── data_processing/               
│   │   ├── download_data.py           # Descarga desde FRED/Yahoo/Shiller
│   │   └── create_monthly.py          # Agregación a frecuencia mensual
│   │
│   ├── models/                        # Modelos econométricos
│   │   ├── vecm_analysis.py           # VECM trivariado (H2, H3)
│   │   ├── reaction_function.py       # Función de reacción Fed (H1)
│   │   ├── hmm_regimes.py             # Hidden Markov Model (regímenes)
│   │   ├── local_projections.py       # Proyecciones locales (H4)
│   │   ├── xgboost_shap.py            # XGBoost + SHAP (H5)
│   │   └── event_study.py             # Event study (QE announcements)
│   │
│   ├── diagnostics/                   # Tests estadísticos
│   │   ├── unit_root_tests.py         # ADF, KPSS (estacionariedad)
│   │   ├── cointegration_tests.py     # Johansen, Gregory-Hansen
│   │   └── residual_diagnostics.py    # Ljung-Box, ARCH-LM, Jarque-Bera
│   │
│   ├── visualization/                 # Gráficos
│   │   ├── time_series_plots.py       # Series en niveles y diferencias
│   │   ├── regime_plots.py            # Visualización HMM
│   │   ├── irf_plots.py               # Impulse Response Functions
│   │   └── shap_plots.py              # SHAP summary/dependence
│   │
│   └── utils/                         # Utilidades
│       ├── config.py                  # Configuración global
│       └── helpers.py                 # Funciones auxiliares
│
├── results/                           # Resultados (regenerables)
│   ├── tables/                        # Tablas CSV/LaTeX
│   ├── figures/                       # Gráficos PNG/PDF
│   └── models/                        # Modelos guardados (.pkl)
│
├── notebooks/                         # Jupyter notebooks (exploración)
│   └── 01_exploratory_analysis.ipynb
│
├── scripts/                           # Scripts ejecutables
│   └── run_all_models.py              # Pipeline completo
│
└── paper/                             # Documento final TFG
    └── TFG_Arquitectura_de_la_Liquidez.pdf
```

---

## 🛠️ Instalación

### Requisitos Previos

- **Python 3.10 o superior**
- **Git** (para clonar el repositorio)
- **Cuenta FRED API** (gratuita): [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)

### Paso 1: Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/tfg-liquidez-arquitectura.git
cd tfg-liquidez-arquitectura
```

### Paso 2: Crear entorno virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Paso 4: Configurar API Key de FRED

Crea un archivo `.env` en la raíz del proyecto:
```bash
# Windows
type nul > .env

# macOS/Linux
touch .env
```

Edita `.env` y añade tu clave:
```
FRED_API_KEY=tu_clave_de_32_caracteres_aqui
```

**⚠️ IMPORTANTE:** El archivo `.env` está en `.gitignore` y NO se subirá a GitHub.

---

## 🚀 Uso Rápido

### Pipeline Completo (recomendado)
```bash
# Ejecutar todos los pasos: descarga → procesamiento → modelos → visualización
python scripts/run_all_models.py
```

### Pasos Individuales
```bash
# 1. Descargar datos desde FRED, Yahoo Finance y Shiller
python src/data_processing/download_data.py

# 2. Crear dataset mensual agregado
python src/data_processing/create_monthly.py

# 3. Generar gráficos exploratorios
python src/visualization/time_series_plots.py

# 4. Tests de raíz unitaria (ADF, KPSS)
python src/diagnostics/unit_root_tests.py

# 5. Estimar VECM
python src/models/vecm_analysis.py

# 6. Identificar regímenes con HMM
python src/models/hmm_regimes.py

# 7. Modelo XGBoost + SHAP
python src/models/xgboost_shap.py
```

### Resultados

Todos los outputs se guardan en `results/`:
- **Tablas:** `results/tables/*.csv`
- **Gráficos:** `results/figures/*.png`
- **Modelos:** `results/models/*.pkl`

---

## 📚 Marco Teórico

### Tesis Central

Las expansiones de liquidez estabilizan transitoriamente los mercados financieros, pero alteran las señales intertemporales que coordinan ahorro e inversión. Si la reducción del coste del capital no refleja un aumento genuino del ahorro real, la estructura productiva resultante puede volverse inconsistente intertemporalmente.

### Canales de Transmisión

1. **Canal directo (portfolio rebalancing):**  
   Fed compra Treasuries → Inversores buscan mayor rentabilidad → Flujo hacia equity

2. **Canal de compresión de primas:**  
   Balance ↑ → Primas de riesgo ↓ → Tasa de descuento ↓ → Valoración ↑

3. **Canal de expectativas (forward guidance):**  
   Compromiso de mantener tipos bajos → Reduce yields largos → Equity más atractivo

4. **Canal de riesgo de cola (tail risk):**  
   Fed como backstop → Reduce percepción de riesgo extremo → Comprime VIX

### Literatura Relacionada

- **Gagnon et al. (2011):** Large-scale asset purchases efectivas en reducir yields
- **Krishnamurthy & Vissing-Jorgensen (2011):** QE comprime primas de término y liquidez
- **Bernanke & Kuttner (2005):** Política monetaria afecta equity principalmente vía primas de riesgo
- **Cieslak & Vissing-Jorgensen (2021):** Fed reaction function responde a equity prices

---

## 🔬 Metodología Empírica

### 1. Vector Error Correction Model (VECM)

**Especificación:**

$$
\Delta X_t = \Pi X_{t-1} + \sum_{i=1}^{p-1} \Gamma_i \Delta X_{t-i} + \varepsilon_t
$$

donde $X_t = (s_t, e_t, b_t)'$ con:
- $s_t = \log(\text{S&P 500}_t)$
- $e_t = \log(\text{Earnings}_t)$
- $b_t = \log(\text{Balance Fed}_t)$

**Tests:**
- Johansen (1995): Rango de cointegración
- Gregory-Hansen (1996): Cointegración con cambio estructural
- Weak exogeneity: $\alpha_b = 0$?

### 2. Hidden Markov Model (HMM)

**Modelo:**

$$
r_t | S_t \sim \mathcal{N}(\mu_{S_t}, \sigma^2_{S_t})
$$

con $S_t \in \{\text{bull}, \text{bear}\}$ y matriz de transición:

$$
P = \begin{pmatrix} p_{11} & 1-p_{11} \\ 1-p_{22} & p_{22} \end{pmatrix}
$$

**Algoritmo:** Expectation-Maximization (Baum-Welch)

### 3. XGBoost + SHAP

**Modelo:**

$$
r_{t+1} = f(\Delta b_t, \Delta v_t, \Delta \sigma_t, \Delta ff_t, \Delta slope_t) + \varepsilon_t
$$

**Interpretación:** Shapley values (SHAP) descomponen predicción en contribuciones aditivas

### 4. Proyecciones Locales

**Modelo (Jordà, 2005):**

$$
r_{t+h} = \alpha_h + \beta_h u_t^{(L)} + \theta_h \left( u_t^{(L)} \cdot \mathbb{1}_{\{S_t=\text{bear}\}} \right) + \gamma_h' Z_t + \varepsilon_{t+h}
$$

**Test:** $H_0: \theta_h = 0$ (efecto homogéneo entre regímenes)

---

## 📊 Datos

### Fuentes de Datos

| Variable | Código | Fuente | Frecuencia Original | Obs. |
|----------|--------|--------|---------------------|------|
| S&P 500 | `^GSPC` | Yahoo Finance | Diaria | 6,283 |
| VIX | `^VIX` | Yahoo Finance | Diaria | 6,282 |
| Balance Fed | `WALCL` | FRED | Semanal | 1,203 |
| Federal Funds Rate | `DFF` | FRED | Diaria | 9,497 |
| Treasury 2Y | `DGS2` | FRED | Diaria | 6,783 |
| Treasury 10Y | `DGS10` | FRED | Diaria | 6,783 |
| Spread BBB | `BAMLC0A4CBBB` | FRED | Diaria | 6,870 |
| PIB Nominal | `GDP` | FRED | Trimestral | 103 |
| Earnings, CAPE | - | Shiller | Mensual | 301 |

### Periodo de Análisis

**Muestra completa:** Enero 2000 - Enero 2025 (301 observaciones mensuales)

**Submuestras:**
- **Pre-QE:** 2000-2008 (crisis tradicional)
- **QE Era:** 2009-2019 (QE1, QE2, QE3, normalización)
- **COVID-QT:** 2020-2025 (QE ilimitado + posterior QT)

### Transformaciones
```python
# Logaritmos (para VECM)
log_sp500 = log(sp500)
log_balance = log(fed_balance)
log_earnings = log(earnings)

# Rendimientos y crecimientos (Δlog ≈ retorno porcentual)
ret_sp500 = Δlog_sp500
growth_balance = Δlog_balance

# Diferencias simples (variables ya en %)
delta_vix = Δvix
delta_spread = Δspread_bbb
delta_ff = Δff_rate

# Pendiente de curva
slope_curve = treasury_10y - treasury_2y
delta_slope = Δslope_curve
```

---

## 📈 Resultados Principales

> **⚠️ NOTA:** Los resultados mostrados son preliminares. El análisis completo está en desarrollo.

### Visualización: Balance Fed vs S&P 500

![Balance vs SP500](results/figures/fig_balance_vs_sp500.png)

**Observación:** Correlación visual positiva entre expansiones del balance (QE) y valoración del S&P 500, especialmente evidente en 2008-2014 y 2020-2021.

### Series en Niveles

![Series Levels](results/figures/fig_series_levels.png)

**Panel S&P 500:** Crecimiento exponencial (escala logarítmica lineal)  
**Panel Balance Fed:** Tres regímenes claros: pre-QE (~800B), QE1-3 (~4.5T), COVID (~9T)  
**Panel VIX:** Spikes en crisis (2008: ~60, 2020: ~80)  
**Panel Spread BBB:** Picos en crisis (2008: ~8 bps, 2020: ~4 bps)

### Rendimientos Mensuales

![Returns](results/figures/fig_series_returns.png)

**Rendimiento S&P 500:** Distribución aproximadamente simétrica con outliers extremos (-18% COVID crash, +12% rebound)  
**Crecimiento Balance:** Barras verdes (QE) concentradas en 2008-2009 y 2020, barras rojas (QT) en 2018-2019 y 2022-2024  
**Cambio VIX:** Alta volatilidad en crisis, baja en periodos tranquilos

---

## 🔄 Replicación

### Replicación Completa (Desde cero)
```bash
# 1. Clonar repositorio
git clone https://github.com/TU_USUARIO/tfg-liquidez-arquitectura.git
cd tfg-liquidez-arquitectura

# 2. Crear entorno
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configurar FRED API key
echo "FRED_API_KEY=tu_clave" > .env

# 4. Ejecutar pipeline completo
python scripts/run_all_models.py

# Tiempo estimado: ~5-10 minutos
# Output: Todas las tablas, gráficos y modelos en results/
```

### Replicación con Datos Incluidos

Si compartes el repositorio con `monthly_data.csv` incluido:
```bash
# Saltar descarga de datos, ir directo a modelos
python src/diagnostics/unit_root_tests.py
python src/models/vecm_analysis.py
python src/models/hmm_regimes.py
python src/models/xgboost_shap.py
```

### Tests Unitarios
```bash
# Ejecutar tests (opcional, si implementaste)
pytest tests/
```

---

## 🤝 Contribuciones

Este es un proyecto académico (TFG), pero se aceptan sugerencias:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit cambios (`git commit -m 'Añade mejora X'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

### Áreas de Mejora Bienvenidas

- [ ] Extensión a otros mercados (Europa, Asia)
- [ ] Modelos adicionales (SVAR, BVAR, TVP-VAR)
- [ ] Dashboard interactivo (Streamlit/Dash)
- [ ] Tests adicionales (Granger causality, cointegración no lineal)

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver archivo [LICENSE](LICENSE) para detalles.
```
MIT License

Copyright (c) 2026 Pablo Álvarez de Toledo Rodríguez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 📧 Contacto

**Pablo Álvarez de Toledo Rodríguez**  
- 📧 Email: [pabloalvarezdetoledo@gmail.com(mailto:pabloalvarezdetoledo@gmail.com)
- 🔗 LinkedIn: www.linkedin.com/in/pablo-alvarez-de-toledo(www.linkedin.com/in/pablo-alvarez-de-toledo)

## 🙏 Agradecimientos

- **FRED (Federal Reserve Bank of St. Louis):** Acceso gratuito a datos macroeconómicos
- **Robert Shiller:** Dataset CAPE de acceso público
- **Yahoo Finance API:** Datos de mercado en tiempo real
- **Comunidad open source:** Statsmodels, scikit-learn, XGBoost, matplotlib

---

## 📚 Referencias Clave

- Bernanke, B. S., & Kuttner, K. N. (2005). "What explains the stock market's reaction to Federal Reserve policy?" *Journal of Finance*, 60(3), 1221-1257.

- Gagnon, J., Raskin, M., Remache, J., & Sack, B. (2011). "The financial market effects of the Federal Reserve's large-scale asset purchases." *International Journal of Central Banking*, 7(1), 3-43.

- Jordà, Ò. (2005). "Estimation and inference of impulse responses by local projections." *American Economic Review*, 95(1), 161-182.

- Krishnamurthy, A., & Vissing-Jorgensen, A. (2011). "The effects of quantitative easing on interest rates: Channels and implications for policy." *Brookings Papers on Economic Activity*, 2011(2), 215-287.

---

## 📊 Badges y Estadísticas

![GitHub last commit](https://img.shields.io/github/last-commit/TU_USUARIO/tfg-liquidez-arquitectura)
![GitHub repo size](https://img.shields.io/github/repo-size/TU_USUARIO/tfg-liquidez-arquitectura)
![GitHub language count](https://img.shields.io/github/languages/count/TU_USUARIO/tfg-liquidez-arquitectura)
![GitHub top language](https://img.shields.io/github/languages/top/TU_USUARIO/tfg-liquidez-arquitectura)

---

**⭐ Si encuentras útil este proyecto, considera darle una estrella en GitHub ⭐**
```

---

## 📝 **ARCHIVOS ADICIONALES RECOMENDADOS**

### **LICENSE**

Crea archivo `LICENSE` en la raíz:
```
MIT License

Copyright (c) 2026 Pablo Álvarez de Toledo Rodríguez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
