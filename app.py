import re
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import phpserialize
import requests
import streamlit as st
from bs4 import BeautifulSoup

log = logging.getLogger(__name__)


# ----------------------------
# Configuration / constants
# ----------------------------
FIS_LIVE_URL = "https://www.fis-ski.com/DB/alpine-skiing/live.html"
HEADERS = {"User-Agent": "MyFISScraper/0.1"}

DISCIPLINE_F = {
    "Giant Slalom": 1010,
    "Slalom": 730,
    "Downhill": 1250,
    "Super G": 1190,
    "Alpine Combined": 1360,
}

PENALTY_BY_CATEGORY = {
    # Race level 0
    "OWG":  (0.00,   0.00),
    "WC":   (0.00,   0.00),
    "WSC":  (0.00,   0.00),
    "COM":  (0.00,   4.00),
    "WQUA": (0.00,   4.00),

    # Race level 1
    "ANC":  (15.00, 999.00),
    "EC":   (15.00, 999.00),
    "ECOM": (15.00, 999.00),
    "FEC":  (15.00, 999.00),
    "NAC":  (15.00, 999.00),
    "SAC":  (15.00, 999.00),
    "UVS":  (15.00, 999.00),
    "WJC":  (15.00, 999.00),
    "EQUA": (23.00, 999.00),

    # Race level 2
    "NC":   (20.00, 999.00),

    # Race level 3
    "AWG":   (23.00, 999.00),
    "CISM":  (23.00, 999.00),
    "CIT":   (40.00, 999.00),
    "CITWC": (40.00, 999.00),
    "CORP":  (23.00, 999.00),
    "EYOF":  (23.00, 999.00),
    "FIS":   (23.00, 999.00),
    "FQUA":  (23.00, 999.00),
    "JUN":   (23.00, 999.00),
    "NJC":   (23.00, 999.00),
    "NJR":   (23.00, 999.00),
    "UNI":   (23.00, 999.00),
    "YOG":   (23.00, 999.00),

    # Race level 4
    "ENL":  (60.00, 999.00),
}


# ----------------------------
# Helpers (mostly your logic)
# ----------------------------
def _looks_like_xml(text: str) -> bool:
    t = text.lstrip("\ufeff \t\r\n")
    return t.startswith("<")


def _try_hosts(
    session: requests.Session,
    hosts: Iterable[str],
    codex: int,
    timeout: tuple[float, float],
) -> Optional[str]:
    codex = str(codex).zfill(4)
    for host in hosts:
        url = f"https://{host}/al{codex}/main.xml"
        try:
            with session.get(url, timeout=timeout) as r:
                if r.status_code == 200 and _looks_like_xml(r.text):
                    return url
        except (requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError):
            continue
    return None


@st.cache_data(ttl=300, show_spinner=False)
def find_working_base_url(codex: int) -> str:
    timeout = (3.05, 10.0)
    sys_hosts = (f"sys{i}.novius.net" for i in range(1, 7))
    d_hosts   = (f"d{i}.novius.net"   for i in range(25, 34))

    with requests.Session() as s:
        url = _try_hosts(s, sys_hosts, codex, timeout)
        if url:
            return url

        url = _try_hosts(s, d_hosts, codex, timeout)
        if url:
            return url

    raise RuntimeError(f"No working sys1..sys6 or d25..d33 for al{codex}")


def fix_php_serialized_utf8(php_bytes: bytes) -> bytes:
    text = php_bytes.decode("utf-8")

    def repl(match: re.Match) -> str:
        content = match.group(2)
        new_len = len(content.encode("utf-8"))
        return f's:{new_len}:"{content}";'

    fixed_text = re.sub(r's:(\d+):"(.*?)";', repl, text, flags=re.DOTALL)
    return fixed_text.encode("utf-8")


@st.cache_data(ttl=120, show_spinner=False)
def fetch_race_data(url: str) -> Dict[str, Any]:
    resp = requests.get(url, headers=HEADERS, timeout=(10, 30))
    resp.raise_for_status()
    raw = resp.content

    m = re.search(br"<lt>(.*)</lt>", raw, flags=re.DOTALL)
    if not m:
        raise ValueError("No <lt>...</lt> block found in response")
    php_bytes = m.group(1).strip()

    php_bytes = fix_php_serialized_utf8(php_bytes)

    data = phpserialize.loads(
        php_bytes,
        decode_strings=True,
    )
    return data


def to_indexed_list(arr):
    if isinstance(arr, list):
        return arr
    if isinstance(arr, dict):
        return [arr[i] for i in sorted(arr.keys())]
    raise TypeError(f"Unexpected array type: {type(arr)}")


def extract_race_data(data, discipline: str, F: float) -> Dict[str, pd.DataFrame]:
    def _clean_value(v, none_sentinels):
        if v in none_sentinels:
            return None
        if isinstance(v, str):
            s = v.strip()
            if s.endswith(":p2"):
                s = s[:-3]
            if s.endswith(":c"):
                s = s[:-2]
            if s.isdigit():
                return int(s)
            return s
        return v

    def normalize_results(dict_list, none_sentinels=("None", None)):
        max_key = max(
            (max(d.keys()) for d in dict_list if isinstance(d, dict) and d),
            default=None
        )
        if max_key is None:
            return [None] * len(dict_list), None

        normalized = []
        for item in dict_list:
            if not isinstance(item, dict) or not item:
                normalized.append(None)
                continue
            v = item.get(max_key, None)
            normalized.append(_clean_value(v, none_sentinels))
        return normalized, max_key

    t_racers = to_indexed_list(data["racers"])
    df_racers = pd.DataFrame(t_racers)
    df_racers.rename(columns={0: "id", 1: "last_name", 2: "first_name", 3: "nation"}, inplace=True)

    t_startlist_run_1 = to_indexed_list(data["startlist"][0][0][0])
    df_startlist_1 = pd.DataFrame(t_startlist_run_1)
    df_startlist_1.rename(columns={0: "id", 1: "bib", 2: "finish"}, inplace=True)

    # Optional run 2
    df_startlist_2 = None
    try:
        t_startlist_run_2 = to_indexed_list(data["startlist"][0][0][1])
        df_startlist_2 = pd.DataFrame(t_startlist_run_2)
        df_startlist_2.rename(columns={0: "id", 1: "bib", 2: "finish"}, inplace=True)
    except Exception:
        t_startlist_run_2 = None

    t_results_run_1 = to_indexed_list(data["result"][0][0])
    results_1_normalized, _ = normalize_results(t_results_run_1)
    df_results_1 = pd.DataFrame(results_1_normalized, columns=["run_1_time"])

    # Optional total
    df_results_total = None
    t_results_total = None
    try:
        t_results_total = to_indexed_list(data["result"][0][1])
        results_total_normalized, _ = normalize_results(t_results_total)
        df_results_total = pd.DataFrame(results_total_normalized, columns=["combined_time"])
    except Exception:
        pass

    leaderboard = None
    df_points = None

    if discipline in ["Downhill", "Super G"]:
        df_run_1 = df_startlist_1.join(df_results_1).merge(df_racers, on="id", how="left")
        df_run_1["run_1_time"] = np.where(df_run_1["finish"] == "finish", df_run_1["run_1_time"], None)

        leaderboard = df_run_1.sort_values(by="run_1_time")
        leaderboard["run_1_time"] = leaderboard["run_1_time"] / 1000
        leaderboard["gap"] = leaderboard["run_1_time"] - leaderboard.iloc[0, 4]
        leaderboard["race_points"] = ((leaderboard["run_1_time"] / leaderboard.iloc[0, 4]) - 1) * F
        leaderboard["final_time"] = leaderboard["run_1_time"]

        df_points = leaderboard[["id", "bib", "nation", "first_name", "last_name", "finish",
                                 "final_time", "gap", "race_points"]].reset_index(drop=True)

    elif discipline in ["Giant Slalom", "Slalom"]:
        if t_results_total is not None and df_results_total is not None and df_startlist_2 is not None:
            df_run_1 = df_startlist_1.join(df_results_1).merge(df_racers, on="id", how="left")
            df_run_2 = df_startlist_2.join(df_results_total).merge(df_racers, on="id", how="left")

            final_result_df = (
                df_run_1.merge(df_run_2, on=["id", "first_name", "last_name", "nation", "bib"], how="left")
                .drop(columns=["3_x", "4_x", "5_x", "6_x", "3_y", "4_y", "5_y", "6_y"])
            )

            # In your original, run_2_time was derived; keep available if needed
            final_result_df["run_2_time"] = final_result_df["combined_time"] - final_result_df["run_1_time"]

            leaderboard = final_result_df.sort_values(by="combined_time")
            leaderboard["combined_time"] = leaderboard["combined_time"] / 1000
            winner_time = leaderboard.iloc[0]["combined_time"]
            leaderboard["gap"] = (leaderboard["combined_time"] - winner_time).round(2)
            leaderboard["race_points"] = ((leaderboard["combined_time"] / winner_time) - 1) * F
            leaderboard["final_time"] = leaderboard["combined_time"]

            # finish logic (mirrors intent of your original)
            finish_x = df_run_1.set_index("id")["finish"]
            finish_y = df_run_2.set_index("id")["finish"]
            leaderboard["finish_x"] = leaderboard["id"].map(finish_x)
            leaderboard["finish_y"] = leaderboard["id"].map(finish_y)

            conditions = [
                (leaderboard["finish_x"] == "finish") & (leaderboard["finish_y"] == "finish"),
                leaderboard["finish_x"] == "dnf",
                leaderboard["finish_y"] == "dnf",
            ]
            results = ["finish", "dnf1", "dnf2"]
            leaderboard["finish"] = np.select(conditions, results, default=None)

            df_points = leaderboard[["id", "bib", "nation", "first_name", "last_name", "finish",
                                     "final_time", "gap", "race_points"]].reset_index(drop=True)
        else:
            # Run 2 not started yet (your fallback logic)
            df_run_1 = df_startlist_1.join(df_results_1).merge(df_racers, on="id", how="left")
            df_run_1["run_1_time"] = np.where(df_run_1["finish"] == "finish", df_run_1["run_1_time"], None)

            leaderboard = df_run_1.sort_values(by="run_1_time")
            leaderboard["run_1_time"] = leaderboard["run_1_time"] / 1000
            leaderboard["gap"] = leaderboard["run_1_time"] - leaderboard.iloc[0, 4]
            leaderboard["race_points"] = ((leaderboard["run_1_time"] / leaderboard.iloc[0, 4]) - 1) * F
            leaderboard["final_time"] = leaderboard["run_1_time"] * 2
            leaderboard["gap"] = leaderboard["gap"] * 2

            df_points = leaderboard[["id", "bib", "nation", "first_name", "last_name", "finish",
                                     "final_time", "gap", "race_points"]].reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported discipline for points: {discipline}")

    return {"leaderboard": leaderboard, "startlist": df_startlist_1, "df_points": df_points}


def to_scalar(x):
    if isinstance(x, np.ndarray):
        return x.item() if x.size == 1 else np.nan
    if isinstance(x, (list, tuple)):
        return x[0] if len(x) == 1 else np.nan
    return x


@st.cache_data(ttl=86400, show_spinner=False)
def retrieve_fispoints(fis_id: int, discipline: str) -> Optional[float]:
    url = (
        "https://www.fis-ski.com/DB/general/athlete-biography.html"
        f"?sectorcode=AL&fiscode={fis_id}&type=fispoints"
    )
    with requests.Session() as s:
        resp = s.get(url, headers=HEADERS, timeout=(10, 30))
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    rows_1 = soup.select_one("div.g-xs-13.g-sm-13.g-md-17.flex-xs-wrap.flex-sm-wrap")
    if rows_1 is None:
        return None

    rows = rows_1.select("div.g-xs-24.g-md")

    records = []
    for row in rows:
        cells = row.select(":scope > div")
        if len(cells) >= 2:
            records.append(cells[1].get_text(strip=True))

    # Mapping aligned with your original indexing assumptions
    points = {
        "SL": records[1] if len(records) > 1 else None,
        "GS": records[2] if len(records) > 2 else None,
        "SG": records[3] if len(records) > 3 else None,
        "DH": records[0] if len(records) > 0 else None,
        "AC": records[4] if len(records) > 4 else None,
    }

    conds = [
        discipline == "Slalom",
        discipline == "Giant Slalom",
        discipline == "Super G",
        discipline == "Downhill",
        discipline == "Alpine Combined",
    ]
    vals = [points["SL"], points["GS"], points["SG"], points["DH"], points["AC"]]
    out = np.select(conds, vals, default=None)

    out = to_scalar(out)
    if out is None:
        return None
    out = str(out).strip()
    return float(out) if out.replace(".", "", 1).isdigit() else None


@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_events(past_hours: int, future_hours: int) -> pd.DataFrame:
    with requests.Session() as s:
        resp = s.get(FIS_LIVE_URL, headers=HEADERS, timeout=(10, 30))
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.select("div.table-row.pointer.reset-padding")

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(hours=past_hours)
    end_utc = now_utc + timedelta(hours=future_hours)

    events = []
    for row in rows:
        onclick = row.get("onclick", "")
        m = re.search(r"fct_open_live\('([^']+)','([^']+)'\)", onclick)
        if not m:
            continue
        live_url, codex = m.groups()

        iso_date = row.select_one("div.timezone-date")["data-iso-date"] if row.select_one("div.timezone-date") else ""
        normal_date = row.select_one("div.timezone-date")["data-date"] if row.select_one("div.timezone-date") else ""

        if not iso_date:
            if not normal_date:
                continue
            event_time_utc = datetime.strptime(normal_date, "%Y-%m-%d").replace(hour=12).astimezone(timezone.utc)
        else:
            event_time_utc = datetime.fromisoformat(iso_date).astimezone(timezone.utc)

        if not (start_utc <= event_time_utc <= end_utc):
            continue

        def get_data(selector):
            el = row.select_one(selector)
            return el.get_text(strip=True) if el else None

        category_el = row.select_one("div.split-row.split-row_bordered")
        category = (
            category_el.select_one("div.split-row__item").get_text(strip=True)
            if category_el and category_el.select_one("div.split-row__item")
            else None
        )

        events.append({
            "date": get_data("div.timezone-date"),
            "live_url": live_url + "l",
            "codex": str(codex.zfill(4)),
            "country": get_data("span.country__name-short"),
            "location": get_data("div.split-row__item.bold"),
            "category": category,
            "gender": get_data("div.gender__inner"),
            "discipline": get_data("div.split-row__item.hidden-xs"),
            "result_status": get_data("div.g-md-5.justify-left.hidden-sm-down"),
            "live": get_data("div.live__content"),
        })

    return pd.DataFrame(events)


def compute_penalty_and_scores(
    df_points: pd.DataFrame,
    startlist: pd.DataFrame,
    leaderboard: pd.DataFrame,
    discipline: str,
    category: str,
    F: float
) -> Tuple[float, float, float, float, pd.DataFrame]:
    # A: best 5 FIS points among top 15 starters
    top_15 = startlist.head(15).copy()
    top_15["points"] = top_15["id"].apply(lambda x: retrieve_fispoints(int(x), discipline))
    top_15["points"] = pd.to_numeric(top_15["points"], errors="coerce")
    A = top_15.sort_values("points")["points"].head(5).sum()

    # B & C: best 5 FIS points among top 10 + best 5 race points among those top 10
    top_10 = leaderboard.head(10).copy()
    top_10["points"] = top_10["id"].apply(lambda x: retrieve_fispoints(int(x), discipline))
    top_10["points"] = pd.to_numeric(top_10["points"], errors="coerce")
    top_10_sorted = top_10.sort_values("points")

    B = top_10_sorted["points"].head(5).sum()
    C = top_10_sorted["race_points"].head(5).sum()

    raw_penalty = (A + B - C) / 10

    if category not in PENALTY_BY_CATEGORY:
        raise ValueError(f"Unknown category code: {category!r}")

    min_penalty, max_penalty = PENALTY_BY_CATEGORY[category]
    penalty = min(max(raw_penalty, min_penalty), max_penalty)

    winners_time = float(df_points.iloc[0]["final_time"])
    pps = float(F) / winners_time

    scored = df_points.copy()
    
    scored["gap"] = pd.to_numeric(scored["gap"], errors="coerce").astype("float64")

    scored["score"] = np.where(scored["gap"] == None, None, (penalty + scored["gap"] * pps).round(2))

    return raw_penalty, penalty, winners_time, pps, scored


# ----------------------------
# Streamlit UI
# ----------------------------



st.set_page_config(page_title="FIS Live Points Calculator", layout="wide")
st.title("FIS Live Points Calculator")

with st.sidebar:
    st.header("Event discovery window (UTC)")
    past_hours = st.slider("Include events from the past (hours)", 1, 72, 24, 1)
    future_hours = st.slider("Include events into the future (hours)", 0, 48, 12, 1)
    st.caption("Tip: widen the window if no events are returned.")
    refresh = st.button("Refresh events")

# Refresh cache on demand
if refresh:
    fetch_live_events.clear()

with st.spinner("Fetching live/recent events..."):
    events_df = fetch_live_events(past_hours=past_hours, future_hours=future_hours)

if events_df.empty:
    st.warning("No events found in the selected time window.")
    st.stop()

st.subheader("Available events")
st.dataframe(events_df, use_container_width=True, hide_index=True)

# Event selection
def event_label(row: pd.Series) -> str:
    return f"{row.get('date', '')} | {row.get('location', '')} | {row.get('gender', '')} | {row.get('discipline', '')} | {row.get('category', '')} | {row.get('result_status', '')}"

labels = [event_label(events_df.iloc[i]) for i in range(len(events_df))]
selected_label = st.selectbox("Select an event", labels, index=0)
selected_idx = labels.index(selected_label)
selected = events_df.iloc[selected_idx].to_dict()

codex = str(selected["codex"])
category = selected.get("category")
discipline = selected.get("discipline")

colA, colB, colC = st.columns(3)
colA.metric("Codex", codex)
colB.metric("Category", category if category else "—")
colC.metric("Discipline", discipline if discipline else "—")

if category == "TRA":
    st.error("Training runs (TRA) do not have FIS points. Select a different event.")
    st.stop()

if discipline not in DISCIPLINE_F:
    st.error(f"Unsupported discipline for points calculation: {discipline!r}")
    st.stop()

F = DISCIPLINE_F[discipline]

run = st.button("Run calculation", type="primary")

if run:
    try:
        with st.spinner("Finding working XML host..."):
            working_url = find_working_base_url(int(codex))
        st.success(f"Working XML: {working_url}")

        with st.spinner("Fetching and parsing race data..."):
            data = fetch_race_data(working_url)

        with st.spinner("Building leaderboard and race points..."):
            output = extract_race_data(data=data, discipline=discipline, F=F)
            leaderboard = output["leaderboard"]
            startlist = output["startlist"]
            df_points = output["df_points"]

        st.subheader("Leaderboard / Race Points")
        st.dataframe(df_points, use_container_width=True, hide_index=True)

        with st.spinner("Computing penalty and final scores (includes athlete FIS points lookups)..."):
            raw_penalty, penalty, winners_time, pps, scored = compute_penalty_and_scores(
                df_points=df_points,
                startlist=startlist,
                leaderboard=leaderboard,
                discipline=discipline,
                category=category,
                F=F,
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Raw penalty", f"{raw_penalty:.2f}")
        m2.metric("Clamped penalty", f"{penalty:.2f}")
        m3.metric("Winner time (s)", f"{winners_time:.2f}")
        m4.metric("Points per second", f"{pps:.4f}")

        st.subheader("Final Scores")
        st.dataframe(scored, use_container_width=True, hide_index=True)

        csv = scored.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scores as CSV",
            data=csv,
            file_name=f"fis_scores_codex_{codex}.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.exception(e)


