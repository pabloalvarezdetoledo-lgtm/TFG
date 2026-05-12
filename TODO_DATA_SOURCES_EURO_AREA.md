# TODO fuentes de datos euro area

Estado despues de Fase 2. No se deben anadir codigos nuevos al descargador sin confirmar que existen, tienen cobertura suficiente y coinciden con la definicion empirica.

## Fuentes confirmadas e implementadas

| Variable objetivo | Estado | Fuente/codigo | Archivo bruto | Cobertura observada en Fase 2 | Decision |
|---|---|---|---|---|---|
| `eurostoxx50` | Confirmada | ECB Data Portal `RTD.M.S0.N.C_DJE50.X` | `data/raw/ecb_eurostoxx50.csv` | 2000-01 a 2026-02 | Usada en dataset mensual base; es promedio mensual de cierres, no cierre de fin de mes |
| `eurostoxx50` | Confirmada parcial | Yahoo `^STOXX50E` | `data/raw/yahoo_eurostoxx50.csv` | 2007-03-30 a 2026-02-27 | Descargada como contraste diario; no se usa como base por cobertura desde 2007 |
| `ciss` | Confirmada | ECB Data Portal `CISS.D.U2.Z0Z.4F.EC.SS_CIN.IDX` | `data/raw/ecb_ciss.csv` | 2000-01-03 a 2026-02-27 | Usada como estres financiero base; mensual por ultimo dato |
| `deposit_facility_rate` | Confirmada | ECB Data Portal `FM.D.U2.EUR.4F.KR.DFR.LEV` | `data/raw/ecb_deposit_facility_rate.csv` | 2000-01-01 a 2026-03-01 | Usada como policy rate base; FRED `ECBDFR` queda como fallback no usado |
| `estr` | Confirmada con muestra corta | ECB Data Portal `EST.B.EU000A2X2A25.WT` | `data/raw/ecb_estr.csv` | 2019-10-01 a 2026-02-27 | Descargada, pero no sirve como baseline full-sample desde 2000 |
| `euro_area_2y_yield` | Confirmada con inicio 2004 | ECB Data Portal `YC.B.U2.EUR.4F.G_N_C.SV_C_YM.SR_2Y` | `data/raw/euro_area_2y_yield.csv` | 2004-09-06 a 2026-02-27 | Usada para curva 2Y/10Y desde 2004; no es Bund aleman |
| `euro_area_10y_yield` | Confirmada con inicio 2004 | ECB Data Portal `YC.B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y` | `data/raw/euro_area_10y_yield.csv` | 2004-09-06 a 2026-02-27 | Usada para curva 2Y/10Y desde 2004; no es Bund aleman |
| `eurosystem_total_assets` | Confirmada | ECB Data Portal `ILM.W.U2.C.T000000.Z5.Z01` | `data/raw/ecb_eurosystem_total_assets.csv` | 2000-W01 a 2026-W09 | Usada como proxy de liquidez full-sample inicial |
| `excess_liquidity` | Confirmada con muestra muy corta | ECB Data Portal `ILM.D.U2.C.EXLIQ.U2.EUR` | `data/raw/ecb_excess_liquidity.csv` | 2024-09-27 a 2026-03-01 | Conceptualmente preferente, pero no baseline full-sample con la serie directa |

## Fuentes parcialmente confirmadas o no implementadas

| Variable objetivo | Estado | Fuente candidata | Decision pendiente |
|---|---|---|---|
| `vstoxx` | Parcial | Yahoo `V2TX.DE` | El ticker existe, pero Yahoo no devuelve historico descargable util. Buscar fuente STOXX/Eurex autorizada o proveedor academico |
| `app_holdings` | Pendiente | ECB Data Portal / OMO / publicaciones APP | Confirmar codigo que represente tenencias APP agregadas; no usar proxy agregado sin etiquetarlo |
| `pepp_holdings` | Pendiente | ECB Data Portal / OMO / publicaciones PEPP | Confirmar codigo que represente tenencias PEPP agregadas |
| `european_credit_spread` | Pendiente | ICE/BofA, ECB, FRED u otra fuente legalmente accesible | Evitar fuentes restringidas; si no hay serie abierta, mantener CISS como stress baseline |
| `emu_growth` | Pendiente | MSCI EMU Growth; Kenneth French Europe fallback | Confirmar acceso/licencia MSCI. Kenneth French Europe solo como fallback Europe, no EMU |
| `emu_value` | Pendiente | MSCI EMU Value; Kenneth French Europe fallback | Confirmar acceso/licencia MSCI. No fabricar serie EMU |
| Fundamentales europeos para VECM | Pendiente | Eurostat, ECB, MSCI, Datastream/Refinitiv si disponible | Decidir si hay earnings/dividendos/valoraciones robustas para VECM como extension |
| Calendario de eventos ECB | Pendiente | ECB press releases, monetary policy decisions | Construir calendario SMP, OMT, APP, PEPP, TLTRO, hikes/cuts de deposito |

## Convencion de CSV tras Fase 2

CSV creados o esperados en `data/raw/`:

- `ecb_eurostoxx50.csv` con columna `eurostoxx50` (base mensual usada)
- `yahoo_eurostoxx50.csv` con columna `eurostoxx50` (contraste diario, cobertura corta)
- `yahoo_vstoxx.csv` con columna `vstoxx` (pendiente; Yahoo no genero historico util)
- `ecb_ciss.csv` con columna `ciss`
- `ecb_deposit_facility_rate.csv` con columna `deposit_facility_rate`
- `ecb_estr.csv` con columna `estr`
- `euro_area_2y_yield.csv` con columna `euro_area_2y_yield`
- `euro_area_10y_yield.csv` con columna `euro_area_10y_yield`
- `ecb_eurosystem_total_assets.csv` con columna `eurosystem_total_assets`
- `ecb_excess_liquidity.csv` con columna `excess_liquidity`
- `ecb_app_holdings.csv` con columna `app_holdings` (pendiente)
- `ecb_pepp_holdings.csv` con columna `pepp_holdings` (pendiente)
- `european_credit_spread.csv` con columna `european_credit_spread` (pendiente)
- `emu_growth.csv` con columna `emu_growth` (pendiente)
- `emu_value.csv` con columna `emu_value` (pendiente)

## Notas metodologicas

- CISS, VSTOXX y otras variables de estres se agregan inicialmente con ultimo dato mensual. Queda como robustez probar promedio mensual.
- Tipos de interes, balance del Eurosistema, liquidez y tenencias de programas se agregan con ultimo dato mensual.
- `eurosystem_total_assets` permite muestra larga; `excess_liquidity` directa no permite todavia un baseline desde 2000.
- La curva 2Y/10Y se basa en la curva ECB para el area del euro, no en Bunds alemanes.
