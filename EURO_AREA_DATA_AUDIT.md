# Auditoria de datos euro area - Fase 2

## Resumen de muestra

- Inicio de muestra mensual: `2000-01-31`
- Fin de muestra mensual: `2026-02-28`
- Observaciones mensuales: `314`
- Soporte 2000-2026 con nucleo minimo usando activos totales: `si`
- Modelo minimo listo con activos totales, CISS, EURO STOXX 50 y deposit facility rate: `si`
- Exceso de liquidez listo como baseline full-sample: `solo muestra corta desde 2024`
- VSTOXX listo: `no`
- Growth/Value listo: `no`
- Curva 2Y/10Y lista: `si, desde 2004`

## Cobertura y missing values

- `eurostoxx50`: 0 missing, primer valido 2000-01-31, ultimo valido 2026-02-28
- `ciss`: 0 missing, primer valido 2000-01-31, ultimo valido 2026-02-28
- `deposit_facility_rate`: 0 missing, primer valido 2000-01-31, ultimo valido 2026-02-28
- `estr`: 237 missing, primer valido 2019-10-31, ultimo valido 2026-02-28
- `euro_area_2y_yield`: 56 missing, primer valido 2004-09-30, ultimo valido 2026-02-28
- `euro_area_10y_yield`: 56 missing, primer valido 2004-09-30, ultimo valido 2026-02-28
- `eurosystem_total_assets`: 0 missing, primer valido 2000-01-31, ultimo valido 2026-02-28
- `excess_liquidity`: 296 missing, primer valido 2024-09-30, ultimo valido 2026-02-28
- `log_eurostoxx50`: 0 missing, primer valido 2000-01-31, ultimo valido 2026-02-28
- `ret_eurostoxx50`: 1 missing, primer valido 2000-02-29, ultimo valido 2026-02-28
- `log_eurosystem_total_assets`: 0 missing, primer valido 2000-01-31, ultimo valido 2026-02-28
- `growth_eurosystem_total_assets`: 1 missing, primer valido 2000-02-29, ultimo valido 2026-02-28
- `log_excess_liquidity`: 296 missing, primer valido 2024-09-30, ultimo valido 2026-02-28
- `growth_excess_liquidity`: 297 missing, primer valido 2024-10-31, ultimo valido 2026-02-28
- `delta_ciss`: 1 missing, primer valido 2000-02-29, ultimo valido 2026-02-28
- `delta_deposit_rate`: 1 missing, primer valido 2000-02-29, ultimo valido 2026-02-28
- `euro_area_slope_curve`: 56 missing, primer valido 2004-09-30, ultimo valido 2026-02-28
- `delta_euro_area_slope`: 57 missing, primer valido 2004-10-31, ultimo valido 2026-02-28

## Decisiones de agregacion

- Precios/indices de mercado: ultimo dato mensual cuando la fuente es diaria; EURO STOXX 50 usa ECB RTD mensual promedio por cobertura 2000-2026.
- CISS y volatilidad: ultimo dato mensual por defecto; queda TODO probar promedio mensual como robustez.
- Tipos de interes: ultimo dato mensual.
- Balance/liquidez/tenencias: ultimo dato mensual.

## Incidencias y pendientes

- `vstoxx`: Ticker V2TX.DE confirmado, pero sin historico descargable util en Fase 2.
- `app_holdings`: Tenencias APP; confirmar definicion antes de descargar.
- `pepp_holdings`: Tenencias PEPP; confirmar definicion antes de descargar.
- `european_credit_spread`: Proxy europeo de credito corporativo pendiente.
- `emu_growth`: MSCI EMU Growth preferente; Kenneth French Europe fallback.
- `emu_value`: MSCI EMU Value preferente; Kenneth French Europe fallback.
- `vstoxx_historical`: Yahoo V2TX.DE existe, pero no devuelve historico descargable; no se fabrica la serie.

Tabla CSV detallada: `results/tables/euro_area_data_audit.csv`
