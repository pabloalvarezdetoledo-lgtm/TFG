# Plan de migracion a euro area - Fase 1

## Alcance

Esta fase inicia la re-especificacion del objeto empirico desde Estados Unidos/Fed hacia euro area/ECB-Eurosistema. No elimina el codigo US: lo conserva como `legacy_us` o `benchmark_us` para reproducibilidad y comparacion secundaria.

Cambios implementados en Fase 1:

- `REGION = "euro_area"` en `src/utils/config.py`.
- Separacion entre `EURO_AREA_SERIES`, `LEGACY_US_SERIES` y `BENCHMARK_US_SERIES`.
- Placeholders explicitos para series europeas cuyo codigo/fuente no esta confirmado.
- Pipeline de descarga capaz de leer la configuracion regional sin romper la descarga US heredada.
- Pipeline mensual con ruta euro area defensiva: avisa si faltan CSV requeridos y continua sin fallar.

No se modifican todavia Local Projections, VAR, VECM, HMM ni XGBoost.

## Inventario de referencias US todavia presentes

| Area | Archivos afectados | Referencias US detectadas | Accion posterior |
|---|---|---|---|
| Configuracion | `src/utils/config.py` | WALCL, DFF, DGS2, DGS10, BAMLC0A4CBBB, RRPONTSYD, WTREGEN, TOTRESNS, `^GSPC`, `^VIX`, Shiller CAPE, eventos Fed/QE | Ya aislado como `legacy_us`/`benchmark_us`; eventos ECB pendientes |
| Descarga | `src/data_processing/download_data.py` | FRED, Yahoo US, Shiller CAPE | Ya parametrizado por region; Shiller omitido del baseline euro area |
| Mensualizacion | `src/data_processing/create_monthly.py` | `sp500`, `vix`, `fed_balance`, `ff_rate`, `treasury_2y`, `treasury_10y`, `spread_bbb`, `rrp_overnight`, `tga`, `total_reserves`, `net_liquidity`, `cape` | Ruta euro area inicial creada; ruta US preservada |
| Diagnosticos | `src/diagnostics/unit_root_tests.py` | `log_sp500`, `log_balance`, `vix`, `ff_rate`, Treasuries, `spread_bbb`, `ret_sp500`, `delta_vix` | Reparametrizar cuando el dataset europeo exista |
| Local Projections | `src/models/Local_projections.py` | `growth_total_reserves`, `ret_sp500`, `delta_spread`, `delta_vix` | Migrar a shocks de liquidez Eurosistema y respuestas euro area |
| VAR | `src/models/VAR_en_Diferencias.py` | `growth_total_reserves`, `ret_sp500`, `delta_vix`, `delta_spread` | Extension posterior; no tocar en Fase 1 |
| VECM | `src/models/VECM_Net_Liquidity.py`, `src/models/VECM_2_NetLiq.py`, `src/models/VECM_Reservas_familiaB.py` | `log_sp500`, `log_earnings`, `log_net_liquidity`, WALCL, reservas | Mantener como legado; VECM europeo solo si hay fundamentales robustos |
| Visualizacion | `src/visualization/time_series_plots.py` | etiquetas S&P 500, Fed, VIX, BBB, `fig_balance_vs_sp500` | Crear visualizaciones euro area en fase posterior |
| Documentacion | `README.md` | Fed, S&P 500, Net Liquidity, Treasury, Shiller | Actualizar narrativa despues de estabilizar datos |
| Datos/resultados | `data/raw`, `data/external`, `data/processed`, `results` | CSV y salidas US existentes | Preservar; no renombrar agresivamente en Fase 1 |

## Mapa de migracion empirica

| Variable US actual | Reemplazo euro area propuesto | Archivos afectados | Transformacion requerida | Incertidumbre de fuente |
|---|---|---|---|---|
| `sp500`, `log_sp500`, `ret_sp500`, S&P 500 | `eurostoxx50`, `log_eurostoxx50`, `ret_eurostoxx50` | `config.py`, `download_data.py`, `create_monthly.py`, diagnosticos, LP, VAR, VECM, visualizacion, README | Yahoo diario a mensual ultimo valor; log; diferencia logaritmica | `^STOXX50E` configurado; confirmar cobertura historica y ajuste |
| `vix`, `delta_vix` | `vstoxx`, `delta_vstoxx` | `config.py`, `download_data.py`, `create_monthly.py`, diagnosticos, LP, VAR, visualizacion | Diario a mensual ultimo valor; diferencia simple | Placeholder `PENDIENTE_CONFIRMAR_VSTOXX` |
| `fed_balance`, `log_balance`, `growth_balance`, WALCL | `eurosystem_total_assets`, `growth_eurosystem_total_assets` | `config.py`, `download_data.py`, `create_monthly.py`, VECM, visualizacion | ECB/Eurosistema a mensual; log si positivo; diferencia logaritmica | Placeholder `PENDIENTE_CONFIRMAR_EUROSYSTEM_TOTAL_ASSETS` |
| `total_reserves`, `growth_total_reserves` | `excess_liquidity`, `growth_excess_liquidity` | `config.py`, `create_monthly.py`, LP, VAR, VECM reservas | ECB/Eurosistema a mensual; log si positivo; diferencia logaritmica | Placeholder `PENDIENTE_CONFIRMAR_EXCESS_LIQUIDITY` |
| `ff_rate`, `delta_ff`, DFF | `deposit_facility_rate`, `delta_deposit_rate`; `estr` como tipo overnight auxiliar | `config.py`, `download_data.py`, `create_monthly.py`, diagnosticos | Tipo oficial a mensual ultimo valor; diferencia simple | Placeholders `PENDIENTE_CONFIRMAR_DEPOSIT_FACILITY_RATE`, `PENDIENTE_CONFIRMAR_ESTR` |
| `treasury_2y`, `treasury_10y`, `slope_curve` | `euro_area_2y_yield`, `euro_area_10y_yield`, `euro_area_slope_curve` | `config.py`, `create_monthly.py`, diagnosticos, VAR | Series de yields a mensual; pendiente 10Y - 2Y; diferencia de pendiente | Placeholders `PENDIENTE_CONFIRMAR_EURO_AREA_2Y_YIELD`, `PENDIENTE_CONFIRMAR_EURO_AREA_10Y_YIELD` |
| `spread_bbb`, `delta_spread` | `ciss`, `delta_ciss`; `european_credit_spread` si es viable | `config.py`, `create_monthly.py`, LP, VAR, visualizacion | CISS/spread a mensual; diferencia simple; usar CISS/VSTOXX como estres base | Placeholders `PENDIENTE_CONFIRMAR_ECB_CISS`, `PENDIENTE_CONFIRMAR_EUROPEAN_CREDIT_SPREAD` |
| `rrp_overnight`, `tga`, `net_liquidity`, `growth_net_liquidity` | Sin equivalente mecanico directo; usar `excess_liquidity`, `eurosystem_total_assets`, `app_holdings`, `pepp_holdings` | `create_monthly.py`, VECM Net Liquidity, visualizacion | No trasladar formula WALCL - RRP - TGA; definir reaccion Eurosistema con variables ECB | Placeholders `PENDIENTE_CONFIRMAR_APP_HOLDINGS`, `PENDIENTE_CONFIRMAR_PEPP_HOLDINGS` |
| Shiller `cape`, `earnings`, `log_earnings` | Retirar del baseline; solo `benchmark_us` o extension europea si hay fundamentales robustos | `download_data.py`, `create_monthly.py`, VECM, diagnosticos, README | No mezclar CAPE US en pipeline europeo principal | Fuente europea pendiente; Shiller queda benchmark US |
| Kenneth French US Growth/Value/HML | `emu_growth`, `emu_value`, `emu_value_minus_growth` | Nueva capa de datos futura, LP/regresiones de valor relativo | Preferir MSCI EMU Growth/Value; fallback Kenneth French Europe claramente etiquetado | Placeholders `PENDIENTE_CONFIRMAR_EMU_GROWTH`, `PENDIENTE_CONFIRMAR_EMU_VALUE` |
| Eventos Fed/QE US | Eventos ECB/Eurosistema: SMP, OMT, APP, PEPP, TLTRO, subidas deposito | `config.py`, visualizacion, event studies futuros | Rehacer calendario institucional europeo | Pendiente de calendario y fechas exactas |

## Prioridad metodologica nueva

1. Funcion de reaccion de liquidez ECB/Eurosistema.
2. Local Projections con shocks de liquidez europea.
3. Local Projections con interacciones de regimen usando CISS o VSTOXX.
4. Growth versus Value relativo con `emu_value_minus_growth`.
5. VAR/VECM solo como extensiones si las series europeas de fundamentales son robustas.

## Proximas fases

1. Confirmar codigos/fuentes en `TODO_DATA_SOURCES_EURO_AREA.md`.
2. Descargar o importar CSV europeos con nombres esperados por `EURO_AREA_MONTHLY_INPUTS`.
3. Ejecutar `create_monthly.py` y validar cobertura/missing values.
4. Reparametrizar diagnosticos y Local Projections.
5. Mantener VECM/VAR como extension, no como baseline inicial.
