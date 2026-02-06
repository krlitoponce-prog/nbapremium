from __future__ import annotations
import os
from collections import defaultdict
import streamlit as st
import requests
import pandas as pd
import sqlite3
import numpy as np
import math
from datetime import datetime, timedelta, date
from bs4 import BeautifulSoup

try:
    from nba_api.stats.endpoints import leaguedashteamstats, teamgamelog
    from nba_api.stats.static import teams
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

# BallDontLie API: poner tu clave en variable de entorno BALLDONTLIE_API_KEY
BALLDONTLIE_BASE = "https://api.balldontlie.io/v1"
# The Odds API: cuotas en vivo. Clave en THE_ODDS_API_KEY (the-odds-api.com, 500 créditos/mes gratis)
THE_ODDS_API_BASE = "https://api.the-odds-api.com/v4"
# Mapeo abreviatura API -> nombre en nuestra app
API_ABBR_TO_OUR_TEAM = {
    "BOS": "Celtics", "OKC": "Thunder", "DEN": "Nuggets", "CLE": "Cavaliers",
    "MIN": "Timberwolves", "NYK": "Knicks", "DAL": "Mavericks", "PHX": "Suns",
    "IND": "Pacers", "LAL": "Lakers", "GSW": "Warriors", "SAC": "Kings",
    "ORL": "Magic", "PHI": "76ers", "NOP": "Pelicans", "MIA": "Heat",
    "HOU": "Rockets", "LAC": "Clippers", "MEM": "Grizzlies", "CHI": "Bulls",
    "ATL": "Hawks", "BKN": "Nets", "TOR": "Raptors", "UTA": "Jazz",
    "SAS": "Spurs", "CHA": "Hornets", "POR": "Trail Blazers", "DET": "Pistons",
    "WAS": "Wizards", "MIL": "Bucks",
}

# Mapeo nombre NBA API / ESPN ("Boston Celtics") -> nuestro key ("Celtics")
NBA_API_TEAM_TO_OUR = {
    "Atlanta Hawks": "Hawks", "Boston Celtics": "Celtics", "Brooklyn Nets": "Nets",
    "Charlotte Hornets": "Hornets", "Chicago Bulls": "Bulls", "Cleveland Cavaliers": "Cavaliers",
    "Dallas Mavericks": "Mavericks", "Denver Nuggets": "Nuggets", "Detroit Pistons": "Pistons",
    "Golden State Warriors": "Warriors", "Houston Rockets": "Rockets", "Indiana Pacers": "Pacers",
    "LA Clippers": "Clippers", "Los Angeles Clippers": "Clippers", "Los Angeles Lakers": "Lakers",
    "Memphis Grizzlies": "Grizzlies", "Miami Heat": "Heat", "Milwaukee Bucks": "Bucks",
    "Minnesota Timberwolves": "Timberwolves", "New Orleans Pelicans": "Pelicans",
    "New York Knicks": "Knicks", "Oklahoma City Thunder": "Thunder", "Orlando Magic": "Magic",
    "Philadelphia 76ers": "76ers", "Phoenix Suns": "Suns", "Portland Trail Blazers": "Trail Blazers",
    "Sacramento Kings": "Kings", "San Antonio Spurs": "Spurs", "Toronto Raptors": "Raptors",
    "Utah Jazz": "Jazz", "Washington Wizards": "Wizards",
}
OUR_TEAM_TO_NBA_API = {v: k for k, v in NBA_API_TEAM_TO_OUR.items()}

# Mapeo nuestros nombres -> The Odds API (formato "Boston Celtics", etc.)
OUR_TEAM_TO_ODDS_API = {
    "Celtics": "Boston Celtics", "Thunder": "Oklahoma City Thunder", "Nuggets": "Denver Nuggets",
    "Cavaliers": "Cleveland Cavaliers", "Timberwolves": "Minnesota Timberwolves", "Knicks": "New York Knicks",
    "Mavericks": "Dallas Mavericks", "Suns": "Phoenix Suns", "Pacers": "Indiana Pacers",
    "Lakers": "Los Angeles Lakers", "Warriors": "Golden State Warriors", "Kings": "Sacramento Kings",
    "Magic": "Orlando Magic", "76ers": "Philadelphia 76ers", "Pelicans": "New Orleans Pelicans",
    "Heat": "Miami Heat", "Rockets": "Houston Rockets", "Clippers": "Los Angeles Clippers",
    "Grizzlies": "Memphis Grizzlies", "Bulls": "Chicago Bulls", "Hawks": "Atlanta Hawks",
    "Nets": "Brooklyn Nets", "Raptors": "Toronto Raptors", "Jazz": "Utah Jazz",
    "Spurs": "San Antonio Spurs", "Hornets": "Charlotte Hornets", "Trail Blazers": "Portland Trail Blazers",
    "Pistons": "Detroit Pistons", "Wizards": "Washington Wizards", "Bucks": "Milwaukee Bucks",
}

# --- 1. CONFIGURACIÓN Y BASE DE DATOS ---
st.set_page_config(page_title="NBA AI PREDICTOR V10.5 FINAL", layout="wide", page_icon="🏀")

def init_db():
    conn = sqlite3.connect('nba_historial.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predicciones 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, fecha TEXT, equipo_l TEXT, equipo_v TEXT,
                  pred_total REAL, casino_total REAL, real_total REAL DEFAULT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS api_team_stats 
                 (equipo TEXT PRIMARY KEY, pts_for REAL, pts_against REAL, updated_at TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- 2. BASES DE DATOS MAESTRAS COMPLETAS (30 EQUIPOS) ---
TEAM_SKILLS = {
    "Celtics": [10, 8, 8, 8], "Thunder": [9, 8, 7, 10], "Nuggets": [8, 10, 9, 7],
    "Cavaliers": [8, 9, 8, 7], "Timberwolves": [7, 9, 10, 7], "Knicks": [8, 8, 9, 6],
    "Mavericks": [10, 7, 7, 8], "Suns": [9, 7, 7, 7], "Pacers": [9, 7, 6, 10],
    "Lakers": [7, 9, 8, 8], "Warriors": [10, 6, 7, 9], "Kings": [8, 8, 7, 9],
    "Magic": [6, 9, 8, 7], "76ers": [8, 8, 8, 7], "Pelicans": [7, 8, 8, 7],
    "Heat": [7, 8, 7, 6], "Rockets": [7, 9, 9, 8], "Clippers": [8, 8, 7, 7],
    "Grizzlies": [7, 8, 8, 8], "Bulls": [7, 7, 7, 8], "Hawks": [8, 7, 7, 9],
    "Nets": [7, 7, 6, 8], "Raptors": [7, 7, 7, 8], "Jazz": [8, 7, 8, 9],
    "Spurs": [7, 8, 7, 8], "Hornets": [7, 6, 7, 8], "Trail Blazers": [6, 7, 7, 7],
    "Pistons": [6, 7, 8, 7], "Wizards": [7, 6, 6, 9], "Bucks": [9, 8, 8, 7]
}

ADVANCED_STATS_FALLBACK = {
    "Celtics": [123.5, 110.5], "Thunder": [119.5, 110.0], "Nuggets": [118.0, 112.0],
    "76ers": [116.5, 113.5], "Cavaliers": [117.2, 110.2], "Lakers": [116.0, 115.0],
    "Warriors": [117.5, 115.8], "Knicks": [118.0, 111.5], "Mavericks": [118.8, 115.2],
    "Bucks": [117.0, 116.2], "Timberwolves": [114.5, 108.2], "Suns": [117.8, 116.0],
    "Pacers": [121.5, 120.0], "Kings": [116.8, 115.5], "Heat": [114.0, 111.8],
    "Magic": [111.5, 109.5], "Clippers": [115.5, 114.0], "Rockets": [113.8, 112.5],
    "Pelicans": [115.0, 113.8], "Hawks": [118.5, 121.2], "Grizzlies": [113.0, 112.8],
    "Bulls": [114.2, 116.5], "Nets": [112.5, 116.8], "Raptors": [113.2, 118.0],
    "Jazz": [115.8, 120.5], "Spurs": [111.0, 115.2], "Hornets": [110.2, 119.5],
    "Pistons": [109.5, 118.0], "Wizards": [111.8, 122.5], "Trail Blazers": [110.0, 117.5]
}

# Equipos de ritmo alto (Ritmo 9-10 en TEAM_SKILLS) -> empujan total hacia arriba
PACE_TEAMS = {"Wizards", "Pacers", "Kings", "Hawks", "Trail Blazers"}

# Peso H2H: cuánto del diff H2H vs proyección aplicamos al total (antes 15%, ahora 40%)
H2H_WEIGHT = 0.40

# Ventaja cancha base; se reduce si el local tiene mal récord
HOME_ADVANTAGE_BASE = 2.5

TEAM_QUARTER_DNA = {
    "Celtics": [0.27, 0.26, 0.24, 0.23], "Thunder": [0.26, 0.26, 0.25, 0.23],
    "Nuggets": [0.25, 0.25, 0.26, 0.24], "76ers": [0.26, 0.25, 0.24, 0.25],
    "Cavaliers": [0.26, 0.26, 0.24, 0.24], "Lakers": [0.24, 0.25, 0.24, 0.27],
    "Warriors": [0.23, 0.24, 0.30, 0.23], "Knicks": [0.25, 0.25, 0.26, 0.24],
    "Mavericks": [0.24, 0.24, 0.25, 0.27], "Bucks": [0.24, 0.25, 0.23, 0.28],
    "Timberwolves": [0.25, 0.26, 0.24, 0.25], "Suns": [0.25, 0.25, 0.25, 0.25],
    "Pacers": [0.28, 0.27, 0.24, 0.21], "Kings": [0.27, 0.26, 0.24, 0.23],
    "Heat": [0.23, 0.24, 0.25, 0.28], "Magic": [0.24, 0.25, 0.26, 0.25],
    "Clippers": [0.25, 0.25, 0.25, 0.25], "Rockets": [0.24, 0.24, 0.26, 0.26],
    "Pelicans": [0.25, 0.26, 0.24, 0.25], "Hawks": [0.27, 0.26, 0.24, 0.23],
    "Grizzlies": [0.24, 0.24, 0.25, 0.27], "Bulls": [0.25, 0.24, 0.24, 0.27],
    "Nets": [0.24, 0.24, 0.24, 0.28], "Raptors": [0.25, 0.25, 0.25, 0.25],
    "Jazz": [0.23, 0.24, 0.26, 0.27], "Spurs": [0.24, 0.24, 0.25, 0.27],
    "Hornets": [0.26, 0.24, 0.24, 0.26], "Pistons": [0.24, 0.24, 0.24, 0.28],
    "Wizards": [0.26, 0.25, 0.23, 0.26], "Trail Blazers": [0.24, 0.24, 0.25, 0.27]
}

def get_matchup_adjusted_dna(team: str, opp_team: str) -> list[float]:
    """
    DNA ajustado al matchup: 85% propio + 15% rival para reflejar dinámicas
    (ej. ante rival Cierrador, Q3/Q4 pueden variar).
    """
    own = TEAM_QUARTER_DNA.get(team, [0.25, 0.25, 0.25, 0.25])
    opp = TEAM_QUARTER_DNA.get(opp_team, [0.25, 0.25, 0.25, 0.25])
    blend = [0.85 * a + 0.15 * b for a, b in zip(own, opp)]
    s = sum(blend)
    return [x / s for x in blend] if s > 0 else own

def project_quarters(total_points: float, dna: list[float]) -> list[float]:
    """
    Proyecta puntos por cuarto asegurando que la suma de Q1–Q4 coincide exactamente
    con el total del equipo y respetando el perfil de arranque/cierre del DNA.
    """
    if not dna or len(dna) != 4:
        weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    else:
        weights = np.array(dna, dtype=float)

    s = weights.sum()
    if s <= 0:
        weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    else:
        weights = weights / s

    # Calculamos Q1–Q3 con los pesos y forzamos Q4 para que cierre el total
    q1 = round(total_points * weights[0], 1)
    q2 = round(total_points * weights[1], 1)
    q3 = round(total_points * weights[2], 1)
    q4 = round(total_points - (q1 + q2 + q3), 1)

    return [q1, q2, q3, q4]


THREE_STAR_PENALTY = 0.12  # Súper estrella: baja ~12% el potencial del equipo
TWO_STAR_PENALTY = 0.06    # Jugador muy importante: baja ~6%
ONE_STAR_PENALTY = 0.00    # Rol / reemplazable: no baja nada

# Nombres en minúsculas para machar con el texto de ESPN
THREE_STAR_PLAYERS = {
    # Súper estrellas, MVP, primeras espadas claras
    "tatum", "jokic", "doncic", "james", "curry", "embiid", "antetokounmpo",
    "davis", "durant", "booker", "gilgeous", "brunson", "wembanayama", "haliburton",
    "lillard", "morant", "butler", "kawhi", "george", "mitchell", "towns",
    "edwards", "irving", "beal", "trae", "fox", "sabonis", "leonard"
}

TWO_STAR_PLAYERS = {
    # Segundas espadas, organizadores, defensores clave
    "brown", "holiday", "porzingis", "murray", "middleton", "maxey", "herro",
    "bane", "jackson", "mobley", "garland", "siakam", "vanvleet", "ayton",
    "gordon", "mikal", "bridges", "mccollum", "ingram", "jamal", "green",
    "poole", "wiggins", "lopez", "allen", "porter", "anunoby", "markkanen",
    "ayton", "dejounte", "white", "reaves", "bane", "simmons"
}

# Mapeo ESPN team names -> nuestros nombres
ESPN_TEAM_TO_OUR = NBA_API_TEAM_TO_OUR  # Mismo formato que NBA API

MONTH_ABBR = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
              "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}


def _parse_return_date(s: str) -> date | None:
    """Parsea '7 Feb', 'Feb 7', 'Apr 1', 'Oct 1' -> date. Año = actual o próximo si ya pasó."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip().lower()
    today = datetime.now().date()
    try:
        parts = s.replace(",", "").split()
        if len(parts) < 2:
            return None
        # Formato "Feb 7" o "7 Feb"
        day = month = None
        for p in parts:
            if p.isdigit():
                day = int(p)
            elif p[:3] in MONTH_ABBR:
                month = MONTH_ABBR[p[:3]]
        if day is None or month is None:
            return None
        year = today.year
        d = datetime(year, month, day).date()
        if d < today and month <= 6:  # fecha pasada en primera mitad año -> próximo año
            d = datetime(year + 1, month, day).date()
        return d
    except (ValueError, KeyError):
        return None


def _fetch_espn_english_injuries():
    """ESPN inglés: nombre, return_date, status (Out/Day-To-Day)."""
    injuries = {}
    try:
        res = requests.get(
            "https://www.espn.com/nba/injuries",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=12,
        )
        soup = BeautifulSoup(res.text, "html.parser")
        for section in soup.find_all("div", class_="ResponsiveTable"):
            title_el = section.find("div", class_="Table__Title")
            if not title_el:
                continue
            team_full = title_el.text.strip()
            our_key = ESPN_TEAM_TO_OUR.get(team_full)
            if not our_key:
                continue
            if our_key not in injuries:
                injuries[our_key] = []
            for row in section.find_all("tr", class_="Table__TR")[1:]:
                tds = row.find_all("td")
                if len(tds) < 4:
                    continue
                name = (tds[0].get_text(strip=True) or "").strip()
                if not name:
                    continue
                ret_str = tds[2].get_text(strip=True) if len(tds) > 2 else ""
                status = (tds[3].get_text(strip=True) or "").lower() if len(tds) > 3 else "out"
                injuries[our_key].append({
                    "name": name,
                    "return_date": _parse_return_date(ret_str),
                    "status": status,
                })
        return injuries
    except Exception:
        return {}


def _espn_deportes_team_to_our(team_raw: str) -> str | None:
    """Mapea nombre equipo ESPN Deportes -> nuestro key."""
    t = team_raw.strip().lower()
    if "76ers" in t or "sixers" in t:
        return "76ers"
    for full, our in ESPN_TEAM_TO_OUR.items():
        if our.lower() in t or full.lower() in t:
            return our
    last = team_raw.split()[-1] if team_raw else ""
    return last if last in TEAM_SKILLS else None


def _fetch_espn_deportes_injuries():
    """ESPN Deportes: nombre, return_date, status si la tabla los tiene."""
    injuries = {}
    try:
        res = requests.get(
            "https://espndeportes.espn.com/basquetbol/nba/lesiones",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        soup = BeautifulSoup(res.text, "html.parser")
        for title in soup.find_all("div", class_="Table__Title"):
            team_raw = title.text.strip()
            team_key = _espn_deportes_team_to_our(team_raw)
            if not team_key:
                continue
            rows_el = title.find_parent("div", class_="ResponsiveTable")
            if not rows_el:
                continue
            rows = rows_el.find_all("tr", class_="Table__TR")[1:]
            if team_key not in injuries:
                injuries[team_key] = []
            for r in rows:
                tds = r.find_all("td")
                name = (tds[0].get_text(strip=True) if tds else "").strip()
                if not name:
                    continue
                ret_str = tds[2].get_text(strip=True) if len(tds) > 2 else ""
                status_raw = (tds[3].get_text(strip=True) or "fuera").lower() if len(tds) > 3 else "fuera"
                status = "out" if "fuera" in status_raw else ("day-to-day" if "día" in status_raw or "dia" in status_raw or "al día" in status_raw else "out")
                injuries[team_key].append({
                    "name": name,
                    "return_date": _parse_return_date(ret_str) if ret_str else None,
                    "status": status,
                })
        return injuries
    except Exception:
        return {}


@st.cache_data(ttl=600)
def get_all_injuries():
    """
    Combina ESPN English + ESPN Deportes. Devuelve {equipo: [{"name", "return_date", "status"}]}.
    El filtrado por fecha (return_date vs game_date) se hace en calculate_injury_penalty.
    """
    eng = _fetch_espn_english_injuries()
    dep = _fetch_espn_deportes_injuries()
    merged = {}
    seen = set()
    for src in (eng, dep):
        for team, players in src.items():
            if team not in merged:
                merged[team] = []
            for p in players:
                key = (team, p["name"].lower())
                if key in seen:
                    continue
                seen.add(key)
                merged[team].append(p)
    return merged


def get_espn_injuries():
    """Legacy: devuelve {equipo: [nombres]} para compatibilidad. Usa get_all_injuries para lógica nueva."""
    all_inj = get_all_injuries()
    return {t: [p["name"] for p in pl] for t, pl in all_inj.items()}


# --- nba_api: stats equipo, home/away, racha, head-to-head ---
def _get_nba_season() -> str:
    """Temporada actual formato '2024-25'."""
    now = datetime.now()
    y = now.year if now.month >= 10 else now.year - 1
    return f"{y}-{str(y+1)[-2:]}"


@st.cache_data(ttl=3600)
def _fetch_nba_api_team_stats():
    """
    nba_api: pts anotados/recibidos por equipo, overall + home + away.
    Devuelve {equipo: {"pts_for": x, "pts_against": x, "pts_home": x, "pts_away": x, "def_home": x, "def_away": x}}.
    """
    if not NBA_API_AVAILABLE:
        return None
    try:
        season = _get_nba_season()
        result = {}
        # Overall
        l_all = leaguedashteamstats.LeagueDashTeamStats(season=season)
        df_all = l_all.get_data_frames()[0]
        # Home
        l_home = leaguedashteamstats.LeagueDashTeamStats(season=season, location_nullable="Home")
        df_home = l_home.get_data_frames()[0]
        # Away
        l_away = leaguedashteamstats.LeagueDashTeamStats(season=season, location_nullable="Away")
        df_away = l_away.get_data_frames()[0]

        for _, row in df_all.iterrows():
            name = row.get("TEAM_NAME", "")
            our_key = NBA_API_TEAM_TO_OUR.get(name)
            if not our_key:
                continue
            result[our_key] = {
                "pts_for": float(row.get("PTS", 0)),
                "pts_against": float(row.get("OPP_PTS", 0)),
            }

        for _, row in df_home.iterrows():
            name = row.get("TEAM_NAME", "")
            our_key = NBA_API_TEAM_TO_OUR.get(name)
            if our_key and our_key in result:
                result[our_key]["pts_home"] = float(row.get("PTS", 0))
                result[our_key]["def_home"] = float(row.get("OPP_PTS", 0))

        for _, row in df_away.iterrows():
            name = row.get("TEAM_NAME", "")
            our_key = NBA_API_TEAM_TO_OUR.get(name)
            if our_key and our_key in result:
                result[our_key]["pts_away"] = float(row.get("PTS", 0))
                result[our_key]["def_away"] = float(row.get("OPP_PTS", 0))

        # Fallback: si no hay home/away, usar overall
        for k, v in result.items():
            if "pts_home" not in v:
                v["pts_home"] = v["pts_for"]
                v["def_home"] = v["pts_against"]
            if "pts_away" not in v:
                v["pts_away"] = v["pts_for"]
                v["def_away"] = v["pts_against"]

        return result if result else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_team_recent_form(team: str, last_n: int = 10):
    """
    Racha reciente (últimos N partidos). Devuelve {"wins": int, "losses": int, "avg_pts": float, "avg_opp_pts": float}.
    """
    if not NBA_API_AVAILABLE:
        return None
    try:
        nba_name = OUR_TEAM_TO_NBA_API.get(team)
        if not nba_name:
            return None
        all_teams = teams.get_teams()
        team_obj = next((t for t in all_teams if t["full_name"] == nba_name), None)
        if not team_obj:
            return None
        tid = team_obj["id"]
        season = _get_nba_season()
        gl = teamgamelog.TeamGameLog(team_id=tid, season=season)
        df = gl.get_data_frames()[0]
        if df.empty or len(df) < 2:
            return None
        df = df.head(last_n)
        wins = int((df["WL"] == "W").sum())
        losses = len(df) - wins
        avg_pts = float(df["PTS"].mean()) if "PTS" in df.columns else 0
        avg_opp = float(df["OPP_PTS"].mean()) if "OPP_PTS" in df.columns else avg_pts  # fallback si no hay
        return {"wins": wins, "losses": losses, "avg_pts": round(avg_pts, 1), "avg_opp_pts": round(avg_opp, 1)}
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_head_to_head(l_team: str, v_team: str, last_n: int = 5):
    """
    Head-to-head: últimos N enfrentamientos. Devuelve {"avg_total_pts": float} o None.
    """
    if not NBA_API_AVAILABLE:
        return None
    try:
        nba_l = OUR_TEAM_TO_NBA_API.get(l_team)
        nba_v = OUR_TEAM_TO_NBA_API.get(v_team)
        if not nba_l or not nba_v:
            return None
        all_teams = teams.get_teams()
        t_l = next((t for t in all_teams if t["full_name"] == nba_l), None)
        t_v = next((t for t in all_teams if t["full_name"] == nba_v), None)
        if not t_l or not t_v:
            return None
        gl = teamgamelog.TeamGameLog(team_id=t_l["id"], season=_get_nba_season())
        df = gl.get_data_frames()[0]
        if df.empty:
            return None
        # MATCHUP contiene "vs. BOS" o "@ BOS" - filtrar por abreviatura del rival
        abbr_v = t_v["abbreviation"]
        mask = df["MATCHUP"].str.contains(abbr_v, case=False, na=False)
        df_h2h = df[mask].head(last_n)
        if df_h2h.empty:
            return None
        # PTS + OPP_PTS = total; si no hay OPP_PTS, estimar ~PTS*2 (asumiendo similar)
        if "OPP_PTS" in df_h2h.columns:
            total_pts = df_h2h["PTS"] + df_h2h["OPP_PTS"]
        else:
            total_pts = df_h2h["PTS"] * 2  # Aproximación
        return {"avg_total_pts": round(float(total_pts.mean()), 1)}
    except Exception:
        return None


def _fetch_balldontlie_team_stats(api_key: str):
    """Obtiene pts anotados y recibidos por equipo desde BallDontLie (temporada actual)."""
    if not api_key or not api_key.strip():
        return None
    headers = {"Authorization": api_key.strip()}
    try:
        # Temporada actual: 2024 = 2024-25
        season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1
        # Listar equipos para tener id -> abreviatura
        r_teams = requests.get(f"{BALLDONTLIE_BASE}/teams", headers=headers, timeout=15)
        if r_teams.status_code != 200:
            return None
        teams_data = r_teams.json()
        if "data" not in teams_data:
            return None
        id_to_abbr = {}
        for t in teams_data["data"]:
            abbr = t.get("abbreviation") or t.get("abbrev")
            if abbr:
                id_to_abbr[str(t["id"])] = abbr

        # Partidos de la temporada (todas las páginas para promedios más precisos)
        all_games = []
        page = 1
        while True:
            r_games = requests.get(
                f"{BALLDONTLIE_BASE}/games",
                headers=headers,
                params={"seasons[]": season, "per_page": 100, "page": page},
                timeout=15,
            )
            if r_games.status_code != 200:
                break
            games_data = r_games.json()
            if "data" not in games_data or not games_data["data"]:
                break
            all_games.extend(games_data["data"])
            if len(games_data["data"]) < 100:
                break
            page += 1
            if page > 50:  # límite de seguridad
                break

        if not all_games:
            return None

        # Acumular por equipo (nuestra clave): listas de pts anotados y recibidos
        pts_for = defaultdict(list)
        pts_against = defaultdict(list)
        for g in all_games:
            home_id = str(g.get("home_team_id"))
            vis_id = str(g.get("visitor_team_id"))
            home_pts = g.get("home_team_score")
            vis_pts = g.get("visitor_team_score")
            if home_pts is None or vis_pts is None:
                continue
            home_abbr = id_to_abbr.get(home_id)
            vis_abbr = id_to_abbr.get(vis_id)
            if home_abbr and home_abbr in API_ABBR_TO_OUR_TEAM:
                our_key = API_ABBR_TO_OUR_TEAM[home_abbr]
                pts_for[our_key].append(home_pts)
                pts_against[our_key].append(vis_pts)
            if vis_abbr and vis_abbr in API_ABBR_TO_OUR_TEAM:
                our_key = API_ABBR_TO_OUR_TEAM[vis_abbr]
                pts_for[our_key].append(vis_pts)
                pts_against[our_key].append(home_pts)

        # Promedios por equipo
        result = {}
        for our_key in API_ABBR_TO_OUR_TEAM.values():
            if our_key in pts_for and pts_for[our_key]:
                result[our_key] = [
                    round(float(np.mean(pts_for[our_key])), 1),
                    round(float(np.mean(pts_against[our_key])), 1),
                ]
        return result if result else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_team_stats_for_prediction():
    """
    Devuelve { equipo: dict con pts_for, pts_against, pts_home, pts_away, def_home, def_away }.
    Prioridad: nba_api > BallDontLie > ADVANCED_STATS_FALLBACK.
    """
    base = {}
    for k in TEAM_SKILLS:
        fallback = ADVANCED_STATS_FALLBACK.get(k, [115, 115])
        base[k] = {"pts_for": fallback[0], "pts_against": fallback[1],
                   "pts_home": fallback[0], "pts_away": fallback[0],
                   "def_home": fallback[1], "def_away": fallback[1]}

    nba_stats = _fetch_nba_api_team_stats()
    if nba_stats:
        for k, v in nba_stats.items():
            base[k] = v
        return base

    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    bdl = _fetch_balldontlie_team_stats(api_key)
    if bdl:
        for k, v in bdl.items():
            base[k] = {"pts_for": v[0], "pts_against": v[1], "pts_home": v[0], "pts_away": v[0],
                       "def_home": v[1], "def_away": v[1]}
    return base


@st.cache_data(ttl=300)
def get_stats_data_source():
    """Indica fuente: nba_api | BallDontLie | fallback."""
    if _fetch_nba_api_team_stats():
        return "nba_api"
    api_key = os.environ.get("BALLDONTLIE_API_KEY")
    if api_key and _fetch_balldontlie_team_stats(api_key):
        return "balldontlie"
    return "fallback"


@st.cache_data(ttl=60)
def fetch_the_odds_api(home_team: str, away_team: str):
    """
    Obtiene cuotas de NBA para el partido local vs visitante.
    Devuelve {"linea": float, "cuota_over": float, "cuota_under": float, "handicap": float, "bookmaker": str}
    o None si no hay API key o no se encuentra el partido.
    """
    api_key = os.environ.get("THE_ODDS_API_KEY")
    if not api_key or not api_key.strip():
        return None
    home_odds = OUR_TEAM_TO_ODDS_API.get(home_team)
    away_odds = OUR_TEAM_TO_ODDS_API.get(away_team)
    if not home_odds or not away_odds:
        return None
    try:
        r = requests.get(
            f"{THE_ODDS_API_BASE}/sports/basketball_nba/odds",
            params={
                "regions": "us,uk,eu",
                "markets": "totals,spreads",
                "oddsFormat": "decimal",
                "apiKey": api_key.strip(),
            },
            timeout=15,
        )
        if r.status_code != 200:
            return None
        games = r.json()
        for g in games:
            h = g.get("home_team", "")
            a = g.get("away_team", "")
            if h != home_odds or a != away_odds:
                continue
            result = {"linea": 220.5, "cuota_over": 1.90, "cuota_under": 1.90, "handicap": -4.5, "bookmaker": ""}
            for bm in g.get("bookmakers", []):
                for m in bm.get("markets", []):
                    if m.get("key") == "totals":
                        for o in m.get("outcomes", []):
                            pt = o.get("point")
                            if pt is not None:
                                result["linea"] = float(pt)
                            if o.get("name") == "Over":
                                result["cuota_over"] = float(o.get("price", 1.90))
                            else:
                                result["cuota_under"] = float(o.get("price", 1.90))
                        result["bookmaker"] = bm.get("title", "")
                        break
                for m in bm.get("markets", []):
                    if m.get("key") == "spreads":
                        for o in m.get("outcomes", []):
                            if o.get("name") == home_odds:
                                pt = o.get("point")
                                if pt is not None:
                                    result["handicap"] = float(pt)
                        break
            return result
        return None
    except Exception:
        return None


def calculate_injury_penalty(team_nick, injuries_db, game_date: date | None = None):
    """
    injuries_db: {equipo: [{"name", "return_date", "status"}]} o legacy {equipo: [nombres]}.
    game_date: fecha del partido. Solo penaliza si OUT y return_date > game_date (o sin fecha).
    Day-To-Day: penalización reducida (50%). Return_date <= game_date: disponible, no penaliza.
    """
    if game_date is None:
        game_date = datetime.now().date()
    raw = injuries_db.get(team_nick, [])
    bajas = []
    for p in raw:
        if isinstance(p, dict):
            name = p.get("name", "")
            ret = p.get("return_date")
            status = (p.get("status") or "out").lower()
            # No penalizar si ya está de vuelta (return_date <= game_date)
            if ret is not None and ret <= game_date:
                continue
            if "day" in status or "dia" in status or "día" in status:
                bajas.append((name, 0.5))  # Day-To-Day: 50% penalty
            elif "out" in status or "fuera" in status:
                bajas.append((name, 1.0))
            else:
                bajas.append((name, 1.0))
        else:
            bajas.append((str(p), 1.0))
    penalty = 0.0
    detected = []
    for p, factor in bajas:
        nombre = p.lower()
        is_star = False

        for s in THREE_STAR_PLAYERS:
            if s in nombre:
                penalty += THREE_STAR_PENALTY * factor
                detected.append(f"⭐⭐⭐ {p}" + (" (D2D)" if factor < 1 else ""))
                is_star = True
                break

        if not is_star:
            for s in TWO_STAR_PLAYERS:
                if s in nombre:
                    penalty += TWO_STAR_PENALTY * factor
                    detected.append(f"⭐⭐ {p}" + (" (D2D)" if factor < 1 else ""))
                    is_star = True
                    break

        if not is_star:
            detected.append(f"⭐ {p}" + (" (D2D)" if factor < 1 else ""))

    return min(0.35, penalty), detected


def get_historial_df():
    conn = sqlite3.connect('nba_historial.db')
    df = pd.read_sql_query("SELECT * FROM predicciones ORDER BY fecha DESC, id DESC", conn)
    conn.close()
    if not df.empty:
        df["pick_modelo"] = np.where(df["pred_total"] > df["casino_total"], "OVER", "UNDER")
        df["pick_real"] = np.where(
            df["real_total"].notna(),
            np.where(df["real_total"] > df["casino_total"], "OVER", "UNDER"),
            None,
        )
        df["acierto"] = (df["pick_modelo"] == df["pick_real"]) & df["real_total"].notna()
    return df


# Calibración: sigma mínimo evita sobreconfianza; solo recomendamos pick si prob > umbral
SIGMA_MIN_PTS = 10.0       # Desviación mínima del error (pts) para no ser demasiado precisos
CONFIDENCE_MIN_PROB = 0.52 # Solo recomendar OVER/UNDER si probabilidad > 52%

def get_error_sigma(df_hist):
    df_valid = df_hist.dropna(subset=["real_total"])
    if df_valid.empty or len(df_valid) < 10:
        return None
    errores = df_valid["real_total"] - df_valid["pred_total"]
    sigma = float(np.std(errores))
    return sigma if sigma > 0 else None


def normal_cdf(x, mu, sigma):
    if sigma is None or sigma <= 0:
        return None
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))


# --- 4. SIDEBAR + LAYOUT ---
with st.sidebar:
    st.title("⚙️ CONTROL ROOM V10.5")
    section = st.radio("Modo", ["Predicción", "Historial/Estadísticas"], index=0)

    if st.button("🔄 REFRESCAR DATOS API"):
        st.rerun()

    # Indicador fuente stats
    stats_src = get_stats_data_source()
    if stats_src == "nba_api":
        st.success("✅ Stats: nba_api (home/away)")
    elif stats_src == "balldontlie":
        st.success("✅ Stats: BallDontLie")
    else:
        st.warning("⚠️ Stats: Fallback estático (instalar nba_api: pip install nba_api)")


    if section == "Predicción":
        st.header("📅 Fecha del partido")
        game_date = st.date_input(
            "Partido para",
            value=datetime.now().date(),
            key="game_date",
            help="Solo se penalizan bajas cuya fecha de regreso sea después de esta fecha.",
        )

        st.header("🦓 Árbitros")
        ref_trend = st.selectbox("Tendencia", ["Neutral", "Over (Pitan Mucho)", "Under (Dejan Jugar)"])

        st.header("🔋 Fatiga (B2B)")
        b2b_local = st.toggle("Local jugó ayer")
        b2b_visita = st.toggle("Visita jugó ayer (+Castigo)")
    else:
        st.header("Filtrado Historial")
        filtro_equipo = st.text_input("Filtrar por equipo (opcional)")


# --- 5. CUERPO PRINCIPAL ---
st.title("🏀 NBA AI PRO: V10.5 FINAL")

if section == "Predicción":
    inj_db = get_all_injuries()
    with st.expander("🚑 REPORTE DE BAJAS E IMPACTO (CLIC PARA ABRIR)"):
        if inj_db:
            for team, players in sorted(inj_db.items()):
                st.markdown(f"**{team}**")
                for p in players:
                    d = p.get("return_date")
                    s = p.get("status", "out")
                    ret_str = f" ({d})" if d else ""
                    st.caption(f"• {p.get('name', p)} — {s}{ret_str}")
        else:
            st.warning("No se pudieron cargar lesiones. Verifique conexión.")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("EQUIPO LOCAL")
        l_team = st.selectbox("Seleccionar Local", sorted(TEAM_SKILLS.keys()), key="local_sel")
        form_l = get_team_recent_form(l_team, 10)
        racha_l = f"Racha: {form_l['wins']}-{form_l['losses']} (últ. 10)" if form_l else "Racha: —"
        st.caption(racha_l)

        pen_l, list_l = calculate_injury_penalty(l_team, inj_db, game_date)
        st.error(f"Impacto Bajas: -{round(pen_l * 100, 1)}% Potencial")
        for p in list_l:
            st.caption(p)

        veng_l_active = st.checkbox("🔥 Factor Venganza Local")

    with col2:
        st.subheader("EQUIPO VISITANTE")
        v_team = st.selectbox("Seleccionar Visitante", sorted(TEAM_SKILLS.keys()), key="visita_sel")
        form_v = get_team_recent_form(v_team, 10)
        racha_v = f"Racha: {form_v['wins']}-{form_v['losses']} (últ. 10)" if form_v else "Racha: —"
        st.caption(racha_v)

        pen_v, list_v = calculate_injury_penalty(v_team, inj_db, game_date)
        st.error(f"Impacto Bajas: -{round(pen_v * 100, 1)}% Potencial")
        for p in list_v:
            st.caption(p)

        veng_v_active = st.checkbox("🔥 Factor Venganza Visita")

    # Línea de referencia para proyección (simplificado, sin Casino)
    if "linea_total_puntos" not in st.session_state:
        st.session_state["linea_total_puntos"] = 220.5
    linea_total_puntos = st.number_input(
        "Línea referencia (pts total)",
        value=220.5,
        key="linea_ref",
        min_value=150.0,
        max_value=300.0,
        step=0.5,
        help="Para comparar proyección OVER/UNDER.",
    )
    st.session_state["linea_total_puntos"] = linea_total_puntos

    if st.button("🚀 EJECUTAR SIMULACIÓN IA", type="primary"):
        team_stats = get_team_stats_for_prediction()
        sl = team_stats.get(l_team)
        sv = team_stats.get(v_team)
        fallback_l = ADVANCED_STATS_FALLBACK.get(l_team, [115, 115])
        fallback_v = ADVANCED_STATS_FALLBACK.get(v_team, [115, 115])

        # Matchup con home/away: local en casa vs defensa visitante fuera; visita fuera vs defensa local en casa
        if sl and sv and isinstance(sl, dict) and isinstance(sv, dict):
            base_score_l = (sl.get("pts_home", sl.get("pts_for", fallback_l[0])) + sv.get("def_away", sv.get("pts_against", fallback_v[1]))) / 2.0
            base_score_v = (sv.get("pts_away", sv.get("pts_for", fallback_v[0])) + sl.get("def_home", sl.get("pts_against", fallback_l[1]))) / 2.0
        else:
            sl = {"pts_for": fallback_l[0], "pts_against": fallback_l[1]}
            sv = {"pts_for": fallback_v[0], "pts_against": fallback_v[1]}
            base_score_l = (sl["pts_for"] + sv["pts_against"]) / 2.0
            base_score_v = (sv["pts_for"] + sl["pts_against"]) / 2.0

        # Modificador racha reciente: si gana mucho, +pts; si pierde mucho, -pts
        form_l = get_team_recent_form(l_team, 10)
        form_v = get_team_recent_form(v_team, 10)
        racha_mod_l = 0.0
        racha_mod_v = 0.0
        if form_l:
            wl = form_l["wins"] - form_l["losses"]
            racha_mod_l = max(-2, min(2, wl * 0.3))  # -2 a +2 pts según racha
        if form_v:
            wl = form_v["wins"] - form_v["losses"]
            racha_mod_v = max(-2, min(2, wl * 0.3))

        # Modificador head-to-head: más peso (40%) cuando H2H difiere mucho
        h2h = get_head_to_head(l_team, v_team, 5)
        h2h_mod_total = 0.0
        if h2h and "avg_total_pts" in h2h:
            proy_base = base_score_l + base_score_v
            diff_h2h = h2h["avg_total_pts"] - proy_base
            # Si H2H difiere mucho (>25 pts), más peso para acercarnos al histórico
            weight = 0.50 if abs(diff_h2h) > 25 else H2H_WEIGHT
            h2h_mod_total = diff_h2h * weight
            h2h_mod_total = max(-10, min(25, h2h_mod_total))  # Cap ampliado para casos extremos

        # Factor ritmo: equipos rápidos suben el total
        pace_mod = 0.0
        if l_team in PACE_TEAMS:
            pace_mod += 2.0
        if v_team in PACE_TEAMS:
            pace_mod += 2.0

        # Ventaja cancha: se reduce si el local tiene mal récord
        home_advantage = HOME_ADVANTAGE_BASE
        if form_l and (form_l["wins"] + form_l["losses"]) >= 5:
            win_pct = form_l["wins"] / (form_l["wins"] + form_l["losses"])
            home_advantage = HOME_ADVANTAGE_BASE * (0.4 + 0.6 * win_pct)  # 0.4-1.0x

        # Modificadores
        ref_mod = 1.02 if "Over" in ref_trend else (0.98 if "Under" in ref_trend else 1.0)
        fat_l = 0.97 if b2b_local else 1.0
        fat_v = 0.96 if b2b_visita else 1.0
        veng_l_val = 1.5 if veng_l_active else 0
        veng_v_val = 1.0 if veng_v_active else 0

        # Cálculo Final (incluye racha + H2H + ritmo + ventaja local ajustada)
        final_l = ((base_score_l * (1 - pen_l) * fat_l) + veng_l_val + home_advantage + racha_mod_l) * ref_mod
        final_v = ((base_score_v * (1 - pen_v) * fat_v) + veng_v_val + racha_mod_v) * ref_mod
        total_ia = round(final_l + final_v + h2h_mod_total + pace_mod, 1)
        spread_ia = round(final_l - final_v, 1)

        # --- MÉTRICAS V9.0 + PROBABILIDADES ---
        st.divider()
        m1, m2, m3 = st.columns([2, 1, 1])

        with m1:
            st.markdown(f"### {l_team} {int(final_l)} - {int(final_v)} {v_team}")
            st.markdown(f"🏆 GANA {'LOCAL' if final_l > final_v else 'VISITA'} por {abs(spread_ia)} pts")
            if h2h and "avg_total_pts" in h2h:
                st.caption(f"📊 H2H últimos 5: {h2h['avg_total_pts']} pts promedio")

        with m2:
            diff_p = round(total_ia - linea_total_puntos, 1)
            st.metric("TOTAL PUNTOS", total_ia, f"{diff_p} vs línea")
            proy_pick = "OVER" if diff_p > 0 else "UNDER"

            # Probabilidades basadas en histórico (sigma mínimo = menos sobreconfianza)
            df_hist_all = get_historial_df()
            sigma_raw = get_error_sigma(df_hist_all) if df_hist_all is not None and not df_hist_all.empty else None
            sigma = max(sigma_raw, SIGMA_MIN_PTS) if sigma_raw is not None else SIGMA_MIN_PTS
            cdf_line = normal_cdf(linea_total_puntos, total_ia, sigma)
            if cdf_line is not None:
                p_over = 1 - cdf_line
                p_under = cdf_line
                st.caption(f"PROB. OVER: {p_over*100:.1f}% | UNDER: {p_under*100:.1f}%")
                # Solo recomendar pick cuando hay ventaja clara (mejor precisión)
                if p_over >= CONFIDENCE_MIN_PROB:
                    st.success(f"✅ RECOMENDACIÓN: OVER (confianza {p_over*100:.0f}%)")
                elif p_under >= CONFIDENCE_MIN_PROB:
                    st.success(f"✅ RECOMENDACIÓN: UNDER (confianza {p_under*100:.0f}%)")
                else:
                    st.warning("⚠️ NEUTRAL: sin ventaja clara — considerar no apostar")
            else:
                st.caption(f"PROYECCIÓN: {proy_pick} (sin calibrar)")

        with m3:
            st.metric("SPREAD PROY.", spread_ia, f"Local por {abs(spread_ia)} pts")

        # --- DESGLOSE POR CUARTOS (mejorado: DNA ajustado al matchup) ---
        st.subheader("🗓️ Desglose por Cuartos")
        dna_l = get_matchup_adjusted_dna(l_team, v_team)
        dna_v = get_matchup_adjusted_dna(v_team, l_team)

        # Perfil: Arrancador (Q1+Q2 > 50%) vs Cierrador (Q3+Q4 > 50%)
        def _perfil(dna):
            arranque = dna[0] + dna[1]
            cierre = dna[2] + dna[3]
            if arranque > 0.51:
                return "🔥 Arrancador", f"Q1+Q2: {arranque*100:.0f}%"
            if cierre > 0.51:
                return "🔚 Cierrador", f"Q3+Q4: {cierre*100:.0f}%"
            return "➖ Equilibrado", f"Q1+Q2: {arranque*100:.0f}%"
        perfil_l, detalle_l = _perfil(dna_l)
        perfil_v, detalle_v = _perfil(dna_v)

        q_l = project_quarters(final_l, dna_l)
        q_v = project_quarters(final_v, dna_v)

        # Tabla con pts y % DNA por cuarto
        df_q = pd.DataFrame({
            "Equipo": [l_team, v_team],
            "Perfil": [perfil_l, perfil_v],
            "Q1": [f"{q_l[0]:.1f} ({dna_l[0]*100:.0f}%)", f"{q_v[0]:.1f} ({dna_v[0]*100:.0f}%)"],
            "Q2": [f"{q_l[1]:.1f} ({dna_l[1]*100:.0f}%)", f"{q_v[1]:.1f} ({dna_v[1]*100:.0f}%)"],
            "Q3": [f"{q_l[2]:.1f} ({dna_l[2]*100:.0f}%)", f"{q_v[2]:.1f} ({dna_v[2]*100:.0f}%)"],
            "Q4": [f"{q_l[3]:.1f} ({dna_l[3]*100:.0f}%)", f"{q_v[3]:.1f} ({dna_v[3]*100:.0f}%)"],
            "TOTAL": [round(final_l, 1), round(final_v, 1)]
        })
        st.table(df_q)
        st.caption(f"{l_team}: {detalle_l} | {v_team}: {detalle_v}")

        # Registro en DB
        conn = sqlite3.connect('nba_historial.db')
        c = conn.cursor()
        c.execute(
            "INSERT INTO predicciones (fecha, equipo_l, equipo_v, pred_total, casino_total) VALUES (?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d"), l_team, v_team, total_ia, linea_total_puntos),
        )
        conn.commit()
        conn.close()

    st.divider()
    st.subheader("🔥 Matchup Heatmap")
    heatmap_modo = st.radio("Ver como", ["Numérico (1-10)", "Porcentaje"], horizontal=True, key="heatmap_modo")
    sk_l = TEAM_SKILLS.get(l_team, [7, 7, 7, 7])
    sk_v = TEAM_SKILLS.get(v_team, [7, 7, 7, 7])
    if "Porcentaje" in heatmap_modo:
        # Escala 6-10 -> 0-100%
        h_l = [round((x - 6) / 4 * 100) for x in sk_l]
        h_v = [round((x - 6) / 4 * 100) for x in sk_v]
    else:
        h_l, h_v = sk_l, sk_v
    h_data = pd.DataFrame({l_team: h_l, v_team: h_v}, index=["Triple", "Pintura", "Rebote", "Ritmo"])
    st.bar_chart(h_data)

    # Pts proyectados por equipo (base matchup)
    team_stats = get_team_stats_for_prediction()
    sl = team_stats.get(l_team)
    sv = team_stats.get(v_team)
    fallback_l = ADVANCED_STATS_FALLBACK.get(l_team, [115, 115])
    fallback_v = ADVANCED_STATS_FALLBACK.get(v_team, [115, 115])
    if sl and sv and isinstance(sl, dict) and isinstance(sv, dict):
        pts_l = round((sl.get("pts_home", sl.get("pts_for", fallback_l[0])) + sv.get("def_away", sv.get("pts_against", fallback_v[1]))) / 2, 1)
        pts_v = round((sv.get("pts_away", sv.get("pts_for", fallback_v[0])) + sl.get("def_home", sl.get("pts_against", fallback_l[1]))) / 2, 1)
    else:
        pts_l = round((fallback_l[0] + fallback_v[1]) / 2, 1)
        pts_v = round((fallback_v[0] + fallback_l[1]) / 2, 1)
    c_pts1, c_pts2 = st.columns(2)
    with c_pts1:
        st.metric(f"{l_team} pts proy. (base)", pts_l)
    with c_pts2:
        st.metric(f"{v_team} pts proy. (base)", pts_v)

else:
    st.subheader("📚 Historial de Predicciones y Estadísticas")
    df_hist = get_historial_df()
    if df_hist is None or df_hist.empty:
        st.info("Aún no hay predicciones registradas. Ejecuta simulaciones en el modo Predicción.")
    else:
        if filtro_equipo:
            mask = df_hist["equipo_l"].str.contains(filtro_equipo, case=False, na=False) | \
                   df_hist["equipo_v"].str.contains(filtro_equipo, case=False, na=False)
            df_hist = df_hist[mask]

        st.dataframe(df_hist)

        df_valid = df_hist.dropna(subset=["real_total"])
        if not df_valid.empty:
            hit_rate = df_valid["acierto"].mean() * 100
            total_preds = len(df_valid)
            st.metric("Hit-Rate OVER/UNDER", f"{hit_rate:.1f} %", help="Basado en picks del modelo vs resultado real")
            st.caption(f"Total picks evaluados: {total_preds}")

            sigma_global = get_error_sigma(get_historial_df())
            if sigma_global is not None:
                st.metric("Desviación típica de error (pts)", f"{sigma_global:.2f}")

            # Gráficos de evolución
            df_valid = df_valid.copy()
            df_valid["fecha_dt"] = pd.to_datetime(df_valid["fecha"], errors="coerce")
            df_valid = df_valid.dropna(subset=["fecha_dt"]).sort_values("fecha_dt")
            df_valid["acierto_int"] = df_valid["acierto"].astype(int)
            df_valid["n"] = np.arange(1, len(df_valid) + 1)
            df_valid["cum_hits"] = df_valid["acierto_int"].cumsum()
            df_valid["cum_hit_rate"] = df_valid["cum_hits"] / df_valid["n"] * 100
            df_valid["error"] = df_valid["real_total"] - df_valid["pred_total"]
            df_valid["abs_error"] = df_valid["error"].abs()

            g1, g2 = st.columns(2)
            with g1:
                st.markdown("#### Evolución Hit-Rate acumulado")
                st.line_chart(df_valid.set_index("fecha_dt")["cum_hit_rate"])
            with g2:
                st.markdown("#### Evolución error absoluto por partido")
                st.line_chart(df_valid.set_index("fecha_dt")["abs_error"])

        st.divider()
        st.subheader("✍️ Registrar resultado real")

        df_pend = df_hist[df_hist["real_total"].isna()]
        if df_pend.empty:
            st.success("No hay predicciones pendientes de resultado.")
        else:
            opciones = {
                f"{row['id']} - {row['fecha']} {row['equipo_l']} vs {row['equipo_v']} (línea {row['casino_total']})": int(row['id'])
                for _, row in df_pend.iterrows()
            }
            etiqueta_sel = st.selectbox("Selecciona el partido", list(opciones.keys()))
            sel_id = opciones[etiqueta_sel]
            real_total_input = st.number_input("Total de puntos real del partido", min_value=0.0, max_value=400.0, value=220.0)
            if st.button("Guardar resultado real"):
                conn = sqlite3.connect('nba_historial.db')
                c = conn.cursor()
                c.execute("UPDATE predicciones SET real_total = ? WHERE id = ?", (real_total_input, sel_id))
                conn.commit()
                conn.close()
                st.success("Resultado guardado correctamente. Recarga para ver las métricas actualizadas.")