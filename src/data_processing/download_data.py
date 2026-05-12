"""
Script de descarga de datos financieros.

La migracion euro area usa REGION desde utils.config. El descargador US queda
disponible mediante REGION="legacy_us" o llamando main("legacy_us").
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from io import StringIO
from urllib.parse import quote

import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import requests
from requests.exceptions import SSLError
from tqdm import tqdm
import urllib3

# Anadir src/ al path para poder importar config
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import (
    DATA_RAW,
    DATA_EXTERNAL,
    START_DATE,
    END_DATE,
    REGION,
    EXTERNAL_SOURCES,
    get_downloadable_series,
    get_region_series,
)


# =============================================================================
# CONFIGURACION INICIAL
# =============================================================================

load_dotenv()
_fred_client = None


def request_with_ssl_fallback(url, headers=None, timeout=60):
    """
    Ejecuta una peticion HTTP con verificacion SSL y fallback documentado.

    En este entorno local la cadena de certificados puede fallar para Yahoo/ECB.
    El fallback no se silencia: devuelve una bandera para que el llamador lo
    imprima y quede trazabilidad en consola/auditoria.
    """
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response, False
    except SSLError as exc:
        print(f"  AVISO SSL: verificacion fallida ({exc}). Reintentando sin verificar certificado.")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(url, headers=headers, timeout=timeout, verify=False)
        response.raise_for_status()
        return response, True


def get_fred_client():
    """Inicializa FRED solo cuando hay series FRED confirmadas que descargar."""
    global _fred_client

    if _fred_client is not None:
        return _fred_client

    fred_api_key = os.getenv("FRED_API_KEY")

    if not fred_api_key:
        raise ValueError(
            "No se encontro FRED_API_KEY en archivo .env\n"
            "Crea un archivo .env en la raiz del proyecto con:\n"
            "FRED_API_KEY=tu_clave_aqui"
        )

    _fred_client = Fred(api_key=fred_api_key)
    return _fred_client


# =============================================================================
# FUNCIONES DE DESCARGA
# =============================================================================

def download_fred_series(series_dict):
    """
    Descarga series desde FRED y las guarda en data/raw/.

    Parameters
    ----------
    series_dict : dict
        Diccionario con formato {'nombre': 'CODIGO_FRED'}.

    Returns
    -------
    dict
        Diccionario con los DataFrames descargados.
    """
    print("\n" + "=" * 70)
    print("DESCARGANDO SERIES DE FRED")
    print("=" * 70)

    if not series_dict:
        print("  No hay series FRED confirmadas para esta region.")
        return {}

    fred = get_fred_client()
    downloaded_data = {}

    for name, fred_code in tqdm(series_dict.items(), desc="Series FRED"):
        try:
            series = fred.get_series(
                fred_code,
                observation_start=START_DATE,
                observation_end=END_DATE,
            )

            df = pd.DataFrame({
                "date": series.index,
                name: series.values,
            })

            info = fred.get_series_info(fred_code)
            frequency = info.get("frequency_short", "Unknown")
            units = info.get("units", "Unknown")

            output_path = DATA_RAW / f"fred_{name}.csv"
            df.to_csv(output_path, index=False)

            downloaded_data[name] = df

            print(
                f"  OK {name:20s} | Codigo: {fred_code:15s} | "
                f"Freq: {frequency:10s} | Units: {units:10s} | Obs: {len(df):5d}"
            )

        except Exception as exc:
            print(f"  ERROR {name:20s} | {exc}")
            continue

    return downloaded_data


def download_yahoo_series(tickers_dict):
    """
    Descarga series desde Yahoo Finance y las guarda en data/raw/.

    Parameters
    ----------
    tickers_dict : dict
        Diccionario con formato {'nombre': 'TICKER'}.

    Returns
    -------
    dict
        Diccionario con los DataFrames descargados.
    """
    print("\n" + "=" * 70)
    print("DESCARGANDO SERIES DE YAHOO FINANCE")
    print("=" * 70)

    if not tickers_dict:
        print("  No hay tickers Yahoo confirmados para esta region.")
        return {}

    downloaded_data = {}

    for name, ticker in tqdm(tickers_dict.items(), desc="Series Yahoo"):
        try:
            start_ts = int(pd.Timestamp(START_DATE, tz="UTC").timestamp())
            end_ts = int(pd.Timestamp(END_DATE, tz="UTC").timestamp())
            encoded_ticker = quote(ticker, safe="")
            url = (
                f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_ticker}"
                f"?period1={start_ts}&period2={end_ts}&interval=1d"
                "&events=history&includeAdjustedClose=true"
            )

            response, insecure_ssl = request_with_ssl_fallback(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=60,
            )
            payload = response.json()
            chart = payload.get("chart", {})
            if chart.get("error"):
                raise ValueError(chart["error"])

            result = (chart.get("result") or [None])[0]
            if not result:
                raise ValueError(f"No se obtuvieron datos para {ticker}")

            timestamps = result.get("timestamp") or []
            indicators = result.get("indicators", {})
            quote_data = (indicators.get("quote") or [{}])[0]
            adjclose_data = (indicators.get("adjclose") or [{}])[0]
            values = adjclose_data.get("adjclose") or quote_data.get("close") or []

            df = pd.DataFrame({
                "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None),
                name: values,
            })
            df = df.reset_index(drop=True).dropna()

            if len(df) == 0:
                raise ValueError(
                    f"Yahoo devolvio metadatos para {ticker}, pero sin observaciones historicas utiles"
                )

            output_path = DATA_RAW / f"yahoo_{name}.csv"
            df.to_csv(output_path, index=False)

            downloaded_data[name] = df

            ssl_note = " | SSL fallback" if insecure_ssl else ""
            print(
                f"  OK {name:20s} | Ticker: {ticker:10s} | Obs: {len(df):5d} | "
                f"{df['date'].min().strftime('%Y-%m-%d')} a {df['date'].max().strftime('%Y-%m-%d')}"
                f"{ssl_note}"
            )

        except Exception as exc:
            print(f"  ERROR {name:20s} | {exc}")
            continue

    return downloaded_data


def parse_ecb_dates(df):
    """Convierte TIME_PERIOD de ECB a fechas pandas defensivamente."""
    periods = df["TIME_PERIOD"].astype(str)
    freq = df["FREQ"].astype(str).iloc[0] if "FREQ" in df.columns and len(df) else ""

    if freq == "W" or periods.str.contains(r"^\d{4}-W\d{2}$", regex=True).any():
        return pd.to_datetime(periods + "-1", format="%G-W%V-%u", errors="coerce")

    if freq == "M" or periods.str.contains(r"^\d{4}-\d{2}$", regex=True).any():
        return pd.to_datetime(periods + "-01", errors="coerce")

    return pd.to_datetime(periods, errors="coerce")


def download_ecb_series(series_dict):
    """
    Descarga series del ECB Data Portal via SDMX CSV.

    series_dict debe contener entradas con flow, key y filename.
    """
    print("\n" + "=" * 70)
    print("DESCARGANDO SERIES DEL ECB DATA PORTAL")
    print("=" * 70)

    if not series_dict:
        print("  No hay series ECB confirmadas para esta region.")
        return {}

    downloaded_data = {}

    for name, meta in tqdm(series_dict.items(), desc="Series ECB"):
        try:
            flow = meta["flow"]
            key = meta["key"]
            filename = meta.get("filename", f"ecb_{name}.csv")
            url = (
                f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}"
                f"?startPeriod={START_DATE}&endPeriod={END_DATE}&format=csvdata"
            )

            response, insecure_ssl = request_with_ssl_fallback(
                url,
                headers={"Accept": "text/csv"},
                timeout=90,
            )

            df_raw = pd.read_csv(StringIO(response.text))
            if df_raw.empty or "OBS_VALUE" not in df_raw.columns or "TIME_PERIOD" not in df_raw.columns:
                raise ValueError("Respuesta ECB vacia o sin columnas OBS_VALUE/TIME_PERIOD")

            df = pd.DataFrame({
                "date": parse_ecb_dates(df_raw),
                name: pd.to_numeric(df_raw["OBS_VALUE"], errors="coerce"),
            })
            df = df.dropna(subset=["date", name]).sort_values("date").reset_index(drop=True)

            if df.empty:
                raise ValueError("Serie ECB sin observaciones numericas validas")

            output_path = DATA_RAW / filename
            df.to_csv(output_path, index=False)
            downloaded_data[name] = df

            freq = df_raw["FREQ"].dropna().iloc[0] if "FREQ" in df_raw and df_raw["FREQ"].notna().any() else "?"
            unit = df_raw["UNIT"].dropna().iloc[0] if "UNIT" in df_raw and df_raw["UNIT"].notna().any() else "?"
            ssl_note = " | SSL fallback" if insecure_ssl else ""
            print(
                f"  OK {name:28s} | {flow}.{key} | Freq: {freq:>2s} | Unit: {unit} | "
                f"Obs: {len(df):5d} | {df['date'].min().strftime('%Y-%m-%d')} a "
                f"{df['date'].max().strftime('%Y-%m-%d')}{ssl_note}"
            )

        except Exception as exc:
            print(f"  ERROR {name:28s} | {exc}")
            continue

    return downloaded_data


def download_shiller_cape(external_sources=None):
    """
    Descarga dataset de Shiller (CAPE, earnings, dividends).

    Shiller CAPE queda como benchmark US y no se llama desde REGION=euro_area.
    """
    sources = external_sources or EXTERNAL_SOURCES

    if "shiller_cape" not in sources:
        print("  Shiller CAPE no esta configurado para esta region.")
        return None

    print("\n" + "=" * 70)
    print("DESCARGANDO DATASET DE SHILLER (CAPE)")
    print("=" * 70)

    try:
        url = sources["shiller_cape"]

        print(f"  Descargando desde: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        temp_file = DATA_EXTERNAL / "shiller_cape_raw.xls"
        with open(temp_file, "wb") as file:
            file.write(response.content)

        try:
            df_raw = pd.read_excel(
                temp_file,
                sheet_name="Data",
                skiprows=7,
                engine="xlrd",
            )
        except ImportError:
            print("  Aviso: xlrd no disponible, intentando lectura alternativa.")
            df_raw = pd.read_excel(
                temp_file,
                sheet_name="Data",
                skiprows=7,
            )

        df = df_raw.iloc[:, [0, 1, 2, 3, 12]].copy()
        df.columns = ["date_raw", "price", "dividend", "earnings", "cape"]

        df["date_raw"] = pd.to_numeric(df["date_raw"], errors="coerce")
        df = df.dropna(subset=["date_raw"])

        df["year"] = df["date_raw"].astype(int)
        df["month"] = ((df["date_raw"] - df["year"]) * 100).round().astype(int)
        df["month"] = df["month"].replace(0, 1)
        df["month"] = df["month"].clip(1, 12)

        df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1), errors="coerce")
        df = df[["date", "price", "dividend", "earnings", "cape"]]

        df = df[
            (df["date"] >= pd.to_datetime(START_DATE)) &
            (df["date"] <= pd.to_datetime(END_DATE))
        ]

        for col in ["price", "dividend", "earnings", "cape"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date"])

        output_path = DATA_EXTERNAL / "shiller_cape.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")

        print(
            f"  OK Shiller CAPE | Obs: {len(df):5d} | "
            f"Periodo: {df['date'].min().strftime('%Y-%m')} a "
            f"{df['date'].max().strftime('%Y-%m')}"
        )

        if temp_file.exists():
            temp_file.unlink()

        return df

    except Exception as exc:
        print(f"  ERROR descargando Shiller CAPE: {exc}")
        print(f"     Tipo de error: {type(exc).__name__}")
        return None


def print_pending_placeholders(series_config):
    """Muestra las series euro area pendientes sin intentar descargarlas."""
    placeholders = series_config.get("placeholders", {})
    if not placeholders:
        return

    print("\n" + "=" * 70)
    print("SERIES PENDIENTES DE CONFIRMACION")
    print("=" * 70)
    for name, meta in placeholders.items():
        print(
            f"  AVISO {name:28s} | Fuente: {meta.get('source', 'pendiente')} | "
            f"Codigo: {meta.get('code', 'PENDIENTE')}"
        )


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main(region=REGION):
    """Ejecuta pipeline de descarga para la region seleccionada."""
    print("\n" + "=" * 70)
    print(" PIPELINE DE DESCARGA DE DATOS FINANCIEROS")
    print("=" * 70)
    print(f"Region activa: {region}")
    print(f"Periodo: {START_DATE} a {END_DATE}")
    print(f"Destino: {DATA_RAW}")
    print("=" * 70)

    start_time = datetime.now()
    series_config = get_region_series(region)

    print_pending_placeholders(series_config)

    ecb_series = series_config.get("ecb", {})
    fred_series = get_downloadable_series(series_config.get("fred", {}))
    yahoo_series = get_downloadable_series(series_config.get("yahoo", {}))
    external_sources = series_config.get("external", {})

    ecb_data = download_ecb_series(ecb_series)
    fred_data = download_fred_series(fred_series)
    yahoo_data = download_yahoo_series(yahoo_series)

    shiller_data = None
    if "shiller_cape" in external_sources:
        shiller_data = download_shiller_cape(external_sources)
    elif region == "euro_area":
        print("\nShiller CAPE omitido: no forma parte del baseline euro area.")

    print("\n" + "=" * 70)
    print("RESUMEN DE DESCARGA")
    print("=" * 70)
    print(f"Series ECB descargadas:   {len(ecb_data)}/{len(ecb_series)}")
    print(f"Series FRED descargadas:  {len(fred_data)}/{len(fred_series)}")
    print(f"Series Yahoo descargadas: {len(yahoo_data)}/{len(yahoo_series)}")
    if "shiller_cape" in external_sources:
        print(f"Shiller CAPE:             {'OK' if shiller_data is not None else 'ERROR'}")
    print(f"Tiempo total: {(datetime.now() - start_time).total_seconds():.1f}s")
    print(f"Archivos guardados en: {DATA_RAW}")
    print("=" * 70)

    print("\nArchivos CSV disponibles:")
    for file in sorted(DATA_RAW.glob("*.csv")):
        size_kb = file.stat().st_size / 1024
        print(f"  - {file.name:40s} ({size_kb:>8.1f} KB)")

    if (DATA_EXTERNAL / "shiller_cape.csv").exists():
        file = DATA_EXTERNAL / "shiller_cape.csv"
        size_kb = file.stat().st_size / 1024
        print(f"  - {file.name:40s} ({size_kb:>8.1f} KB)")

    print("\nDescarga completada\n")


# =============================================================================
# EJECUCION
# =============================================================================

if __name__ == "__main__":
    selected_region = sys.argv[1] if len(sys.argv) > 1 else REGION
    main(selected_region)
