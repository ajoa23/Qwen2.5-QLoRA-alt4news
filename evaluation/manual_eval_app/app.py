import streamlit as st
import pandas as pd
import sqlite3
import requests
from PIL import Image
from io import BytesIO
from itertools import zip_longest

# --- PAGE CONFIG muss als erstes stehen ---
st.set_page_config(layout="wide")

# ---------------------------------------------------------------------------
# 2) GLOBAL CSS  (keine Boxen mehr – nur leichte Abstände)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      div[data-testid="stColumn"]>div:first-child {
            overflow-y: scroll;
            max-height: 70vh;
      }
      div[data-testid="stMainBlockContainer"] {
            max-height: 100vh;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- EINSTELLUNGEN ---
CSV_FILE = "manual_eval_pairwise.csv"
DB_FILE  = "eval_detailed.db"

# --- SQLITE-DB EINRICHTEN ---
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS detailed_evals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_id TEXT,
    model_variant TEXT,
    visibility_principle    INTEGER,
    context_relevance       INTEGER,
    entity_naming           INTEGER,
    informativeness         INTEGER,
    redundancy_avoidance    INTEGER,
    style_readability       INTEGER,
    total                   INTEGER,
    justification           TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(image_id, model_variant)
)
""")
conn.commit()

# --- DATEN LADEN UND CACHE ---
@st.cache_data
def load_df():
    return pd.read_csv(CSV_FILE)

df = load_df()
image_ids = sorted(df["image_id"].unique())

# --- SESSIONSTATE INIT ---
if "idx" not in st.session_state:
    st.session_state.idx = 0

# --- FRAGEN-RENDERING -----------------------------------------------
def render_questions(variant_prefix: str, defaults: dict):
    """
    Zeichnet die sechs Qualitätsfragen + Gesamteindruck zweispaltig ohne Boxen.
    Die Frage steht direkt im Label des Radio-Widgets.
    """
    questions = [
        ("Entspricht der Alt-Text ausschließlich dem visuell Erkennbaren?", "visibility_principle"),
        ("Werden redaktionelle Informationen sinnvoll und korrekt eingebunden?", "context_relevance"),
        ("Sind zentrale Objekte, Personen oder Orte korrekt benannt?", "entity_naming"),
        ("Bietet der Alt-Text spezifischen, bildbezogenen Mehrwert?", "informativeness"),
        ("Verzichtet der Text auf Wiederholungen aus Headline/Caption?", "redundancy_avoidance"),
        ("Ist der Text klar, präzise und verständlich formuliert?", "style_readability"),
        ("Gesamteindruck (1 schlecht … 5 sehr gut)", "total"),
    ]

    left_col, right_col = st.columns(2, gap="small")
    for idx, (q_text, q_key) in enumerate(questions):
        col = left_col if idx % 2 == 0 else right_col
        with col:
            st.radio(
                label=q_text,
                options=[1, 2, 3, 4, 5],
                index=defaults[q_key] - 1,
                key=f"{q_key}_{variant_prefix}",
                horizontal=True,
                label_visibility="visible",
            )

# --- ANNOTATE-FUNKTION -----------------------------------------------
def annotate(image_id, variant):
    entry = df[(df["image_id"] == image_id) & (df["model_variant"] == variant)].iloc[0]

    st.subheader(f"{variant.title()} Alt-Text")
    st.markdown(f"> **{entry['alt_text']}**")
    st.markdown("---")

    # Lade Defaults aus DB oder verwende Mittelwert = 4
    cur = conn.execute("""
       SELECT visibility_principle, context_relevance, entity_naming,
              informativeness, redundancy_avoidance, style_readability,
              total, justification
       FROM detailed_evals
       WHERE image_id=? AND model_variant=?
    """, (image_id, variant))
    row = cur.fetchone()
    defaults = {
        "visibility_principle": row[0] if row else 4,
        "context_relevance":    row[1] if row else 4,
        "entity_naming":        row[2] if row else 4,
        "informativeness":      row[3] if row else 4,
        "redundancy_avoidance": row[4] if row else 4,
        "style_readability":    row[5] if row else 4,
        "total":                row[6] if row else 4,
        "justification":        row[7] if row else ""
    }

    # Fragen rendern
    render_questions(variant, defaults)

    # Freitext-Feld direkt anschließen, ohne zusätzlichen Abstand
    justification = st.text_area(
        "Begründung / Kommentar",
        value=defaults["justification"],
        key=f"just_{variant}",
        height=120,
    )

    # Speichern-Button
    if st.button(f"💾 {variant.title()} speichern", key=f"save_{variant}"):
        conn.execute("DELETE FROM detailed_evals WHERE image_id=? AND model_variant=?", (image_id, variant))
        conn.execute("""
          INSERT INTO detailed_evals (
            image_id, model_variant,
            visibility_principle, context_relevance, entity_naming,
            informativeness, redundancy_avoidance, style_readability,
            total, justification
          ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
          image_id, variant,
          st.session_state[f"visibility_principle_{variant}"],
          st.session_state[f"context_relevance_{variant}"],
          st.session_state[f"entity_naming_{variant}"],
          st.session_state[f"informativeness_{variant}"],
          st.session_state[f"redundancy_avoidance_{variant}"],
          st.session_state[f"style_readability_{variant}"],
          st.session_state[f"total_{variant}"],
          justification
        ))
        conn.commit()
        st.success(f"{variant.title()} gespeichert!")

# --- HAUPT-UI ---------------------------------------------------------
st.title("🔍 Manuelle Qualitätsbewertung von Alt-Texten")
st.markdown("Vergleiche Baseline vs. Fine-Tuned auf sechs Dimensionen plus Gesamteindruck.")

# Layout: Bild + Kontext in linke Spalte (scrollbar), Bewertung in rechte Spalte (scrollbar)
left, right = st.columns((1, 2), gap="large")

with left:
    cid = image_ids[st.session_state.idx]
    row = df[df["image_id"] == cid].iloc[0]
    try:
        resp = requests.get(row["image_url_clean"], timeout=3)
        img  = Image.open(BytesIO(resp.content))
        st.image(img, caption="Originalbild", use_container_width=True)
    except:
        st.warning("Bild konnte nicht geladen werden.")
    st.markdown("---")
    st.markdown(f"**Headline:** {row['headline']}")
    st.markdown(f"**Abstract:** {row['abstract']}")
    st.markdown(f"**Caption:** {row['caption']}")
    st.markdown("---")

    col_back, col_next = st.columns(2, gap="small")
    with col_back:
        if st.button("⬅️ Zurück") and st.session_state.idx > 0:
            st.session_state.idx -= 1
    with col_next:
        if st.button("➡️ Weiter") and st.session_state.idx < len(image_ids)-1:
            st.session_state.idx += 1

    st.markdown(f"{st.session_state.idx+1} / {len(image_ids)}")

with right:
    annotate(cid, "baseline")
    st.markdown("---")
    annotate(cid, "finetuned")
