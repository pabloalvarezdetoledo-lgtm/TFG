"""Configuracion global del proyecto.

Esta primera capa de migracion separa explicitamente el nuevo nucleo euro area
del codigo heredado de Estados Unidos. Las series US se conservan como legado
o benchmark, pero REGION fija el objeto empirico principal en el area del euro.
"""
from pathlib import Path


# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
RESULTS_TABLES = PROJECT_ROOT / "results" / "tables"
RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"
RESULTS_MODELS = PROJECT_ROOT / "results" / "models"

for path in [DATA_PROCESSED, DATA_EXTERNAL, DATA_RAW, RESULTS_TABLES, RESULTS_FIGURES, RESULTS_MODELS]:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Arquitectura regional
# ---------------------------------------------------------------------------
REGION = "euro_area"
VALID_REGIONS = ("euro_area", "legacy_us", "benchmark_us")


# Series heredadas de Estados Unidos. Se mantienen para reproducibilidad y para
# comparaciones benchmark, pero ya no son el nucleo empirico por defecto.
LEGACY_US_FRED_SERIES = {
    "fed_balance": "WALCL",                   # Activos totales de la Reserva Federal
    "ff_rate": "DFF",                         # Federal Funds Effective Rate
    "treasury_2y": "DGS2",                    # Treasury Constant Maturity 2Y
    "treasury_10y": "DGS10",                  # Treasury Constant Maturity 10Y
    "spread_bbb": "BAMLC0A4CBBB",             # ICE BofA BBB US Corporate Index OAS
    "gdp_nominal": "GDP",                     # PIB nominal de EEUU (trimestral)
    "rrp_overnight": "RRPONTSYD",             # Overnight Reverse Repo Facility
    "tga": "WTREGEN",                         # Treasury General Account
    "total_reserves": "TOTRESNS",             # Reservas totales
}

LEGACY_US_YAHOO_TICKERS = {
    "sp500": "^GSPC",                         # S&P 500 Index
    "vix": "^VIX",                            # CBOE Volatility Index
}

LEGACY_US_EXTERNAL_SOURCES = {
    "shiller_cape": (
        "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/"
        "downloads/25d6827d-c04b-447a-bb6d-918d5d88be49/ie_data.xls?ver=1770307872442"
    )
}

LEGACY_US_SERIES = {
    "fred": LEGACY_US_FRED_SERIES,
    "yahoo": LEGACY_US_YAHOO_TICKERS,
    "external": LEGACY_US_EXTERNAL_SOURCES,
}


# Benchmark US opcional. Shiller CAPE queda aqui, no en el pipeline europeo base.
BENCHMARK_US_SERIES = {
    "fred": {
        "ff_rate": LEGACY_US_FRED_SERIES["ff_rate"],
        "treasury_2y": LEGACY_US_FRED_SERIES["treasury_2y"],
        "treasury_10y": LEGACY_US_FRED_SERIES["treasury_10y"],
        "spread_bbb": LEGACY_US_FRED_SERIES["spread_bbb"],
    },
    "yahoo": {
        "sp500": LEGACY_US_YAHOO_TICKERS["sp500"],
        "vix": LEGACY_US_YAHOO_TICKERS["vix"],
    },
    "external": {
        "shiller_cape": LEGACY_US_EXTERNAL_SOURCES["shiller_cape"],
    },
}


# Series euro area descargables desde Yahoo. Para EURO STOXX 50 se conserva
# como fuente de contraste diaria; el dataset mensual base usa la serie RTD del
# ECB porque cubre 2000-2026. VSTOXX queda parcialmente confirmado en Yahoo:
# el ticker existe, pero no devuelve historico util en las pruebas de Fase 2.
EURO_AREA_YAHOO_TICKERS = {
    "eurostoxx50": "^STOXX50E",                # EURO STOXX 50; Yahoo empieza aprox. en 2007
    "vstoxx": "V2TX.DE",                       # VSTOXX; ticker sin historico util via Yahoo
}

EURO_AREA_FRED_SERIES = {
    # FRED queda como fallback opcional, no como fuente preferente euro area.
    # "deposit_facility_rate": "ECBDFR",
}

EURO_AREA_ECB_SERIES = {
    "eurostoxx50": {
        "flow": "RTD",
        "key": "M.S0.N.C_DJE50.X",
        "filename": "ecb_eurostoxx50.csv",
        "frequency": "M",
        "confirmed": True,
        "notes": "EURO STOXX 50 mensual; ECB RTD publica promedio mensual, no cierre de fin de mes.",
    },
    "ciss": {
        "flow": "CISS",
        "key": "D.U2.Z0Z.4F.EC.SS_CIN.IDX",
        "filename": "ecb_ciss.csv",
        "frequency": "D",
        "confirmed": True,
        "notes": "New Composite Indicator of Systemic Stress, euro area.",
    },
    "deposit_facility_rate": {
        "flow": "FM",
        "key": "D.U2.EUR.4F.KR.DFR.LEV",
        "filename": "ecb_deposit_facility_rate.csv",
        "frequency": "D",
        "confirmed": True,
        "notes": "ECB deposit facility rate; FRED ECBDFR queda como fallback no usado.",
    },
    "estr": {
        "flow": "EST",
        "key": "B.EU000A2X2A25.WT",
        "filename": "ecb_estr.csv",
        "frequency": "B",
        "confirmed": True,
        "notes": "Euro short-term rate; empieza en 2019 y no sirve como baseline full-sample desde 2000.",
    },
    "euro_area_2y_yield": {
        "flow": "YC",
        "key": "B.U2.EUR.4F.G_N_C.SV_C_YM.SR_2Y",
        "filename": "euro_area_2y_yield.csv",
        "frequency": "B",
        "confirmed": True,
        "notes": "ECB euro area all-ratings government yield curve, spot rate 2Y; empieza en 2004.",
    },
    "euro_area_10y_yield": {
        "flow": "YC",
        "key": "B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y",
        "filename": "euro_area_10y_yield.csv",
        "frequency": "B",
        "confirmed": True,
        "notes": "ECB euro area all-ratings government yield curve, spot rate 10Y; empieza en 2004.",
    },
    "eurosystem_total_assets": {
        "flow": "ILM",
        "key": "W.U2.C.T000000.Z5.Z01",
        "filename": "ecb_eurosystem_total_assets.csv",
        "frequency": "W",
        "confirmed": True,
        "notes": "Activos/pasivos totales del Eurosistema, millones de euros, frecuencia semanal.",
    },
    "excess_liquidity": {
        "flow": "ILM",
        "key": "D.U2.C.EXLIQ.U2.EUR",
        "filename": "ecb_excess_liquidity.csv",
        "frequency": "D",
        "confirmed": True,
        "notes": "Exceso de liquidez del Eurosistema; serie directa confirmada, pero solo desde 2024-09 en la API.",
    },
}

EURO_AREA_EXTERNAL_SOURCES = {
    # Pendiente: MSCI EMU Growth/Value y spreads corporativos europeos requieren
    # confirmar acceso/licencia antes de automatizar descarga.
}

EURO_AREA_PLACEHOLDER_SERIES = {
    "vstoxx_historical": {
        "source": "STOXX/Eurex u otra fuente autorizada",
        "code": "PENDIENTE_CONFIRMAR_VSTOXX_HISTORICO",
        "notes": "Yahoo V2TX.DE existe, pero no devuelve historico descargable; no se fabrica la serie.",
    },
    "app_holdings": {
        "source": "ECB SDW",
        "code": "PENDIENTE_CONFIRMAR_APP_HOLDINGS",
        "notes": "Confirmar definicion y agregacion de tenencias APP.",
    },
    "pepp_holdings": {
        "source": "ECB SDW",
        "code": "PENDIENTE_CONFIRMAR_PEPP_HOLDINGS",
        "notes": "Confirmar definicion y agregacion de tenencias PEPP.",
    },
    "european_credit_spread": {
        "source": "ICE/BofA/FRED/ECB",
        "code": "PENDIENTE_CONFIRMAR_EUROPEAN_CREDIT_SPREAD",
        "notes": "Elegir spread corporativo europeo comparable a BBB/OAS.",
    },
    "emu_growth": {
        "source": "MSCI o Kenneth French Europe fallback",
        "code": "PENDIENTE_CONFIRMAR_EMU_GROWTH",
        "notes": "Preferir MSCI EMU Growth; Kenneth French Europe solo como proxy.",
    },
    "emu_value": {
        "source": "MSCI o Kenneth French Europe fallback",
        "code": "PENDIENTE_CONFIRMAR_EMU_VALUE",
        "notes": "Preferir MSCI EMU Value; Kenneth French Europe solo como proxy.",
    },
}

EURO_AREA_SERIES = {
    "ecb": EURO_AREA_ECB_SERIES,
    "fred": EURO_AREA_FRED_SERIES,
    "yahoo": EURO_AREA_YAHOO_TICKERS,
    "external": EURO_AREA_EXTERNAL_SOURCES,
    "placeholders": EURO_AREA_PLACEHOLDER_SERIES,
}


# Entradas esperadas por el pipeline mensual europeo. Los CSV pendientes se
# tratan defensivamente: se avisa y se continua sin romper la ejecucion.
EURO_AREA_MONTHLY_INPUTS = {
    "eurostoxx50": {
        "filename": "ecb_eurostoxx50.csv",
        "method": "last",
        "required": True,
        "confirmed": True,
        "source": "ECB RTD",
        "source_frequency": "M",
        "notes": "Indice de renta variable base; la serie ECB RTD es promedio mensual, no cierre mensual.",
    },
    "vstoxx": {
        "filename": "yahoo_vstoxx.csv",
        "method": "last",
        "required": False,
        "confirmed": False,
        "source": "Yahoo (parcial)",
        "source_frequency": "D",
        "notes": "Ticker V2TX.DE confirmado, pero sin historico descargable util en Fase 2.",
    },
    "ciss": {
        "filename": "ecb_ciss.csv",
        "method": "last",
        "required": True,
        "confirmed": True,
        "source": "ECB CISS",
        "source_frequency": "D",
        "notes": "Agregacion mensual por ultimo dato; probar promedio mensual como robustez.",
    },
    "deposit_facility_rate": {
        "filename": "ecb_deposit_facility_rate.csv",
        "method": "last",
        "required": True,
        "confirmed": True,
        "source": "ECB FM",
        "source_frequency": "D",
        "notes": "Sustituye a Fed Funds Rate.",
    },
    "estr": {
        "filename": "ecb_estr.csv",
        "method": "last",
        "required": False,
        "confirmed": True,
        "source": "ECB EST",
        "source_frequency": "B",
        "notes": "Tipo overnight de referencia; revisar tramo pre-2019.",
    },
    "euro_area_2y_yield": {
        "filename": "euro_area_2y_yield.csv",
        "method": "last",
        "required": True,
        "confirmed": True,
        "source": "ECB YC",
        "source_frequency": "B",
        "notes": "Sustituye a Treasury 2Y.",
    },
    "euro_area_10y_yield": {
        "filename": "euro_area_10y_yield.csv",
        "method": "last",
        "required": True,
        "confirmed": True,
        "source": "ECB YC",
        "source_frequency": "B",
        "notes": "Sustituye a Treasury 10Y.",
    },
    "eurosystem_total_assets": {
        "filename": "ecb_eurosystem_total_assets.csv",
        "method": "last",
        "required": True,
        "confirmed": True,
        "source": "ECB ILM",
        "source_frequency": "W",
        "notes": "Sustituye a fed_balance/WALCL.",
    },
    "excess_liquidity": {
        "filename": "ecb_excess_liquidity.csv",
        "method": "last",
        "required": False,
        "confirmed": True,
        "source": "ECB ILM",
        "source_frequency": "D",
        "notes": "Proxy preferente conceptualmente, pero la serie directa empieza en 2024-09.",
    },
    "app_holdings": {
        "filename": "ecb_app_holdings.csv",
        "method": "last",
        "required": False,
        "confirmed": False,
        "notes": "Tenencias APP; confirmar definicion antes de descargar.",
    },
    "pepp_holdings": {
        "filename": "ecb_pepp_holdings.csv",
        "method": "last",
        "required": False,
        "confirmed": False,
        "notes": "Tenencias PEPP; confirmar definicion antes de descargar.",
    },
    "european_credit_spread": {
        "filename": "european_credit_spread.csv",
        "method": "last",
        "required": False,
        "confirmed": False,
        "notes": "Proxy europeo de credito corporativo pendiente.",
    },
    "emu_growth": {
        "filename": "emu_growth.csv",
        "method": "last",
        "required": False,
        "confirmed": False,
        "notes": "MSCI EMU Growth preferente; Kenneth French Europe fallback.",
    },
    "emu_value": {
        "filename": "emu_value.csv",
        "method": "last",
        "required": False,
        "confirmed": False,
        "notes": "MSCI EMU Value preferente; Kenneth French Europe fallback.",
    },
}

EURO_AREA_DERIVED_VARIABLES = [
    "log_eurostoxx50",
    "ret_eurostoxx50",
    "delta_vstoxx",
    "delta_ciss",
    "delta_deposit_rate",
    "euro_area_slope_curve",
    "delta_euro_area_slope",
    "growth_eurosystem_total_assets",
    "growth_excess_liquidity",
    "emu_value_minus_growth",
]


# Aliases de compatibilidad: los scripts heredados que importan FRED_SERIES,
# YAHOO_TICKERS o EXTERNAL_SOURCES siguen apuntando al bloque US legado.
FRED_SERIES = LEGACY_US_FRED_SERIES
YAHOO_TICKERS = LEGACY_US_YAHOO_TICKERS
EXTERNAL_SOURCES = LEGACY_US_EXTERNAL_SOURCES


def get_region_series(region=REGION):
    """Devuelve la configuracion de fuentes para una region."""
    if region == "euro_area":
        return EURO_AREA_SERIES
    if region == "legacy_us":
        return LEGACY_US_SERIES
    if region == "benchmark_us":
        return BENCHMARK_US_SERIES
    raise ValueError(f"Region no reconocida: {region}")


def is_placeholder_code(code):
    """Detecta codigos pendientes para evitar descargas con series inventadas."""
    if code is None:
        return True
    value = str(code).strip().upper()
    return value == "" or value.startswith(("PENDIENTE_", "TODO_", "PLACEHOLDER_"))


def get_downloadable_series(series_dict):
    """Filtra un diccionario de series dejando solo codigos confirmados."""
    return {
        name: code
        for name, code in series_dict.items()
        if not is_placeholder_code(code)
    }


# Parametros de analisis
START_DATE = "2000-01-01"
END_DATE = "2026-03-01"
MONTHLY_FREQ = "M"

# Configuracion VECM
VECM_LAG_ORDER = 2
VECM_DET_ORDER = 0

# Configuracion HMM
HMM_N_STATES = 2
HMM_N_ITTER = 1000
HMM_RANDOM_SEED = 42
HMM_COVARIANCE_TYPE = "full"

# Configuracion XGBoost
XGBOOST_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample": 0.8,
    "random_state": 42,
    "objective": "reg:squarederror",
}

# Configuracion de validacion cruzada para XGBoost
XGBOOST_CV_FOLD = 5
XGBOOST_TEST_SIZE = 24

# Eventos US heredados. Deben moverse a benchmark_US o reemplazarse por eventos
# ECB/Eurosistema antes de reactivar event studies europeos.
EVENTS = {
    "QE1_announcement": "2008-11-25",
    "QE2_announcement": "2010-11-03",
    "Operation_Twist": "2011-09-21",
    "QE3_announcement": "2012-09-13",
    "Taper_tantrum": "2013-05-22",
    "Taper_begins": "2013-12-18",
    "COVID_crisis": "2020-03-11",
    "COVID_QE_unlimited": "2020-03-15",
    "First_rate_hike": "2022-03-16",
    "SVB_collapse": "2023-03-10",
}

# Parametros de visualizacion
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# Colores regimenes HMM
REGIMES_COLORS = {
    "bear": "#d62728",
    "bull": "#2ca02c",
}


def validate_config():
    """
    Valida que la configuracion sea consistente.
    Se ejecuta al importar el modulo.
    """
    from datetime import datetime

    if REGION not in VALID_REGIONS:
        raise ValueError(f"REGION debe estar en {VALID_REGIONS}")

    try:
        datetime.strptime(START_DATE, "%Y-%m-%d")
        datetime.strptime(END_DATE, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Formato de fecha invalido: {exc}") from exc

    if START_DATE >= END_DATE:
        raise ValueError("END_DATE debe ser posterior a START_DATE")

    if HMM_N_STATES < 2:
        raise ValueError("HMM debe tener al menos 2 estados")

    if VECM_LAG_ORDER < 1:
        raise ValueError("VECM debe tener al menos 1 lag en diferencias")

    required_inputs = [
        name for name, meta in EURO_AREA_MONTHLY_INPUTS.items()
        if meta.get("required")
    ]
    if REGION == "euro_area" and "eurostoxx50" not in required_inputs:
        raise ValueError("eurostoxx50 debe ser entrada requerida del pipeline euro area")

    print("Configuracion validada correctamente")


# Ejecutar validacion al importar
validate_config()
