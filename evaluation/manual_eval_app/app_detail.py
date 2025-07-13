import streamlit as st
import pandas as pd
import sqlite3
import requests
from PIL import Image
from io import BytesIO

# --------------------------------------------------
# 1) PAGE CONFIG – muss immer zuerst kommen
# --------------------------------------------------
st.set_page_config(layout="wide")

# --------------------------------------------------
# 2) GLOBAL CSS  (dezentes Layout, Scrollbar in Spalten)
# --------------------------------------------------
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

# --------------------------------------------------
# 3) EINSTELLUNGEN & PERSISTENZ
# --------------------------------------------------
CSV_FILE = "manual_eval_pairwise.csv"     # Datensatz mit beiden Varianten
DB_FILE  = "eval_detailed.db"             # SQLite‑DB für Bewertungen

# -------- SQLite‑DB anlegen (falls nicht vorhanden) --------
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
conn.execute(
    """
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
    """
)
conn.commit()

# -------- Daten laden & (einmalig) mischen --------
@st.cache_data
def load_df():
    # Shuffle, damit die Reihenfolge pro Session randomisiert ist
    return pd.read_csv(CSV_FILE).sample(frac=1, random_state=42).reset_index(drop=True)

df = load_df()
num_entries = len(df)

# --------------------------------------------------
# 4) SESSION‑STATE
# --------------------------------------------------
if "idx" not in st.session_state:
    st.session_state.idx = 0       # Zeigt auf den aktuellen Datensatz‑Eintrag

# --------------------------------------------------
# 5) Hilfs‑Funktionen
# --------------------------------------------------

def render_questions(suffix: str, defaults: dict):
    """Zeichnet die sieben Radio‑Fragen zweispaltig ohne Umrandung."""
    questions = [
        ("Entspricht der Alt‑Text ausschließlich dem visuell Erkennbaren?", "visibility_principle"),
        ("Werden redaktionelle Informationen sinnvoll und korrekt eingebunden?", "context_relevance"),
        ("Sind zentrale Objekte, Personen oder Orte korrekt benannt?", "entity_naming"),
        ("Bietet der Alt‑Text spezifischen, bildbezogenen Mehrwert?", "informativeness"),
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
                key=f"{q_key}_{suffix}",
                horizontal=True,
                label_visibility="visible",
            )


def annotate(row):
    """Rendern eines einzelnen Alt‑Textes inklusive Bewertungs‑Widgets."""
    image_id      = row["image_id"]
    model_variant = row["model_variant"]  # bleibt intern erhalten, wird aber nicht angezeigt
    suffix        = f"{image_id}_{model_variant}"  # sorgt für eindeutige Keys

    st.subheader("Alt‑Text")
    st.markdown(f"> **{row['alt_text']}**")
    st.markdown("---")

    # --- Defaults holen (4 = Mittelwert) ---
    cur = conn.execute(
        """
        SELECT visibility_principle, context_relevance, entity_naming,
               informativeness, redundancy_avoidance, style_readability,
               total, justification
        FROM detailed_evals
        WHERE image_id=? AND model_variant=?
        """,
        (image_id, model_variant),
    )
    res = cur.fetchone()
    defaults = {
        "visibility_principle": res[0] if res else 4,
        "context_relevance":    res[1] if res else 4,
        "entity_naming":        res[2] if res else 4,
        "informativeness":      res[3] if res else 4,
        "redundancy_avoidance": res[4] if res else 4,
        "style_readability":    res[5] if res else 4,
        "total":                res[6] if res else 4,
        "justification":        res[7] if res else "",
    }

    # --- Fragen rendern ---
    render_questions(suffix, defaults)

    # --- Freitext ---
    justification = st.text_area(
        "Begründung / Kommentar",
        value=defaults["justification"],
        key=f"just_{suffix}",
        height=120,
    )

    # --- Speichern‑Button ---
    if st.button("💾 Speichern", key=f"save_{suffix}"):
        conn.execute(
            "DELETE FROM detailed_evals WHERE image_id=? AND model_variant=?",
            (image_id, model_variant),
        )
        conn.execute(
            """
            INSERT INTO detailed_evals (
              image_id, model_variant,
              visibility_principle, context_relevance, entity_naming,
              informativeness, redundancy_avoidance, style_readability,
              total, justification
            ) VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (
                image_id,
                model_variant,
                st.session_state[f"visibility_principle_{suffix}"],
                st.session_state[f"context_relevance_{suffix}"],
                st.session_state[f"entity_naming_{suffix}"],
                st.session_state[f"informativeness_{suffix}"],
                st.session_state[f"redundancy_avoidance_{suffix}"],
                st.session_state[f"style_readability_{suffix}"],
                st.session_state[f"total_{suffix}"],
                justification,
            ),
        )
        conn.commit()
        st.success("Bewertung gespeichert!")

# --------------------------------------------------
# 6) HAUPT‑UI
# --------------------------------------------------

st.title("🔍 Manuelle Qualitätsbewertung von Alt‑Texten")

left, right = st.columns((1, 2), gap="large")

# -------- LINKER BEREICH: Bild & Kontext --------
with left:
    row = df.iloc[st.session_state.idx]
    image_id = row["image_id"]

    # Bild laden
    try:
        resp = requests.get(row["image_url_clean"], timeout=3)
        img  = Image.open(BytesIO(resp.content))
        st.image(img, caption="Originalbild", use_container_width=True)
    except Exception:
        st.warning("Bild konnte nicht geladen werden.")

    st.markdown("---")
    st.markdown(f"**Headline:** {row['headline']}")
    st.markdown(f"**Abstract:** {row['abstract']}")
    st.markdown(f"**Caption:** {row['caption']}")
    st.markdown("---")

    # Navigation
    col_back, col_next = st.columns(2, gap="small")
    with col_back:
        if st.button("⬅️ Zurück") and st.session_state.idx > 0:
            st.session_state.idx -= 1
    with col_next:
        if st.button("➡️ Weiter") and st.session_state.idx < num_entries - 1:
            st.session_state.idx += 1

    st.markdown(f"Eintrag {st.session_state.idx + 1} / {num_entries}")

# -------- RECHTER BEREICH: Bewertung --------
with right:
    annotate(row)