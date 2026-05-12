# Preparacion de modelos euro area - Fase 3

## Estado del dataset

- Dataset usado: `data/processed/monthly_data_euro_area.csv`
- Muestra mensual: `2000-01-31` a `2026-02-28`
- Observaciones mensuales: `314`
- Objeto empirico: area del euro y ECB/Eurosistema.
- Variables US: solo legado o benchmark; no entran en esta preparacion.

## Decision central de liquidez

- Proxy baseline full-sample: `growth_eurosystem_total_assets`.
- Motivo: `eurosystem_total_assets` cubre 2000-2026 sin missing mensuales.
- `excess_liquidity` queda como robustez/descriptiva de muestra corta porque la serie directa empieza en 2024-09.
- No se traslada mecanicamente la formula US `WALCL - RRP - TGA`.

## Nota sobre EURO STOXX 50

- El baseline `eurostoxx50` viene del ECB Data Portal (`RTD.M.S0.N.C_DJE50.X`).
- Esa fuente es una serie mensual de promedio de observaciones del periodo, no cierre de fin de mes.
- Yahoo `^STOXX50E` queda como robustez de muestra corta desde 2007, no como baseline 2000-2026.

## Cobertura de variables clave

- `eurostoxx50`: obs=314, missing=0, primer valido `2000-01-31`, ultimo valido `2026-02-28`.
- `log_eurostoxx50`: obs=314, missing=0, primer valido `2000-01-31`, ultimo valido `2026-02-28`.
- `ret_eurostoxx50`: obs=313, missing=1, primer valido `2000-02-29`, ultimo valido `2026-02-28`.
- `ciss`: obs=314, missing=0, primer valido `2000-01-31`, ultimo valido `2026-02-28`.
- `delta_ciss`: obs=313, missing=1, primer valido `2000-02-29`, ultimo valido `2026-02-28`.
- `deposit_facility_rate`: obs=314, missing=0, primer valido `2000-01-31`, ultimo valido `2026-02-28`.
- `delta_deposit_rate`: obs=313, missing=1, primer valido `2000-02-29`, ultimo valido `2026-02-28`.
- `eurosystem_total_assets`: obs=314, missing=0, primer valido `2000-01-31`, ultimo valido `2026-02-28`.
- `log_eurosystem_total_assets`: obs=314, missing=0, primer valido `2000-01-31`, ultimo valido `2026-02-28`.
- `growth_eurosystem_total_assets`: obs=313, missing=1, primer valido `2000-02-29`, ultimo valido `2026-02-28`.
- `estr`: obs=77, missing=237, primer valido `2019-10-31`, ultimo valido `2026-02-28`.
- `euro_area_2y_yield`: obs=258, missing=56, primer valido `2004-09-30`, ultimo valido `2026-02-28`.
- `euro_area_10y_yield`: obs=258, missing=56, primer valido `2004-09-30`, ultimo valido `2026-02-28`.
- `euro_area_slope_curve`: obs=258, missing=56, primer valido `2004-09-30`, ultimo valido `2026-02-28`.
- `delta_euro_area_slope`: obs=257, missing=57, primer valido `2004-10-31`, ultimo valido `2026-02-28`.
- `excess_liquidity`: obs=18, missing=296, primer valido `2024-09-30`, ultimo valido `2026-02-28`.
- `vstoxx`: obs=0, missing=314, primer valido `n/a`, ultimo valido `n/a`.
- `emu_growth`: obs=0, missing=314, primer valido `n/a`, ultimo valido `n/a`.
- `emu_value`: obs=0, missing=314, primer valido `n/a`, ultimo valido `n/a`.
- `emu_value_minus_growth`: obs=0, missing=314, primer valido `n/a`, ultimo valido `n/a`.
- `app_holdings`: obs=0, missing=314, primer valido `n/a`, ultimo valido `n/a`.
- `pepp_holdings`: obs=0, missing=314, primer valido `n/a`, ultimo valido `n/a`.
- `european_credit_spread`: obs=0, missing=314, primer valido `n/a`, ultimo valido `n/a`.

## Diagnosticos de estacionariedad

Los tests ADF y KPSS se usan solo como guia de especificacion. Si discrepan, se reporta ambiguedad.

- `log_eurostoxx50`: evidencia_no_estacionaria (ADF p=0.2624, KPSS p=0.0162)
- `ret_eurostoxx50`: evidencia_estacionaria (ADF p=0.0000, KPSS p=0.1000)
- `ciss`: evidencia_estacionaria (ADF p=0.0017, KPSS p=0.1000)
- `delta_ciss`: evidencia_estacionaria (ADF p=0.0000, KPSS p=0.1000)
- `deposit_facility_rate`: evidencia_no_estacionaria (ADF p=0.1083, KPSS p=0.0229)
- `delta_deposit_rate`: evidencia_estacionaria (ADF p=0.0000, KPSS p=0.1000)
- `log_eurosystem_total_assets`: evidencia_no_estacionaria (ADF p=0.7757, KPSS p=0.0100)
- `growth_eurosystem_total_assets`: evidencia_estacionaria (ADF p=0.0000, KPSS p=0.1000)
- `euro_area_slope_curve`: evidencia_no_estacionaria (ADF p=0.3530, KPSS p=0.0306)
- `delta_euro_area_slope`: evidencia_estacionaria (ADF p=0.0000, KPSS p=0.1000)

## Inspeccion lead-lag

Convencion: lag negativo significa que la variable financiera lidera a la liquidez; lag positivo significa que la liquidez lidera a la variable financiera.

- `ret_eurostoxx50`: mayor correlacion absoluta en lag -3 (-0.208); uso descriptivo, no causal.
- `delta_ciss`: mayor correlacion absoluta en lag 0 (0.125); uso descriptivo, no causal.
- `delta_deposit_rate`: mayor correlacion absoluta en lag 3 (-0.233); uso descriptivo, no causal.
- `ciss`: mayor correlacion absoluta en lag 1 (0.222); uso descriptivo, no causal.

## Recomendacion para Fase 4

Primera funcion de reaccion recomendada:

```text
growth_eurosystem_total_assets_t = alpha
  + sum beta_i growth_eurosystem_total_assets_{t-i}
  + sum gamma_i ciss_{t-i}
  + sum delta_i ret_eurostoxx50_{t-i}
  + sum theta_i delta_deposit_rate_{t-i}
  + error_t
```

Primera especificacion Local Projection recomendada:

```text
y_{t+h} - y_{t-1} = alpha_h + beta_h shock_liquidity_t + controls_t + error_{t+h}
```

Respuestas candidatas: `ret_eurostoxx50`, `delta_ciss`, `ciss`, `delta_deposit_rate`.

## Estado de readiness

- Baseline Phase 4 listo: `si`
- Curva 2Y/10Y lista como extension desde 2004: `si`
- Exceso de liquidez listo para full-sample: `no`
- Growth/Value listo: `no`
- VSTOXX listo: `no`

## Advertencias metodologicas

- No se debe interpretar la correlacion lead-lag como causalidad.
- No se deben diferenciar variables mecanicamente si se pierde la interpretacion teorica.
- Los modelos definitivos deben controlar la endogeneidad temporal y reportar robustez de rezagos.
- VSTOXX, Growth/Value, APP/PEPP y spreads corporativos siguen no disponibles o no validados.
