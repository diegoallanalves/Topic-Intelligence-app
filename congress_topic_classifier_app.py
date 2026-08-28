from __future__ import annotations

import re
from pathlib import Path
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIG
# ============================================================
TOPICS: Dict[str, str] = {
    "Forced Labor & Xinjiang": (
        "forced labor forced labour Xinjiang Uyghur UFLPA import bans entity lists "
        "supply chain forced labor disclosure cotton polysilicon"
    ),
    "Hong Kong": (
        "Hong Kong autonomy certification separate customs status export control status sanctions officials"
    ),
    "Taiwan": (
        "Taiwan relations arms sales international participation trade investment agreements UNGA Resolution 2758"
    ),
    "Human Rights & Political Repression": (
        "Tibet organ harvesting religious freedom genocide political prisoners censorship Congressional Gold Medals "
        "transnational repression human rights democracy dissidents"
    ),
    "COVID-19 & Pandemic Accountability": (
        "COVID coronavirus pandemic origins Wuhan WHO pandemic liability PPE medical supply dependence"
    ),
    "Narcotics & Fentanyl": (
        "fentanyl precursor scheduling trafficking cartels narcotics opioids drug money laundering controlled substances"
    ),
    "Technology Competition, Telecom & Strategic Sectors": (
        "Huawei ZTE 5G telecom network equipment undersea cables equipment authorization covered entity lists "
        "semiconductors chips quantum artificial intelligence AI biotech drones aerospace technical standards"
    ),
    "Energy, Grid & Electrical Infrastructure": (
        "bulk power system transformers substations switchgear transmission generation solar renewables nuclear oil gas "
        "critical minerals metals electrical grid energy infrastructure"
    ),
    "Foreign Investment in the US": (
        "CFIUS foreign ownership acquire acquisition farmland real estate strategic assets joint ventures "
        "foreign owned firms US programs investment screening"
    ),
    "Information & Technology Security": (
        "intellectual property IP trade secret theft counterfeiting academic research security Confucius Institutes "
        "talent programs espionage counterintelligence foreign agents cyber intrusion personal data apps platforms TikTok"
    ),
    "Sanctions & Export Controls": (
        "sanctions asset freezes entity designation entity list denial orders export licensing dual use controls "
        "export controls restrictions BIS OFAC embargo"
    ),
    "Tariffs, Trade Remedies & Market Access": (
        "tariffs duties customs de minimis Section 301 antidumping countervailing duties normal trade relations "
        "market economy developing country status unfair trade subsidies USTR market access trade remedies"
    ),
    "Supply Chain Security & Reshoring": (
        "critical supply chain reviews reshoring onshoring nearshoring incentives relocation incentives domestic manufacturing "
        "industrial base stockpiles supply chain resilience"
    ),
    "Federal Procurement & Buy American": (
        "federal procurement government contractors buy american domestic content federal acquisition rules public works "
        "procurement prohibition contracting purchasing"
    ),
    "Defense, Military & Territorial Disputes": (
        "defense defence military authorization posture PLA weapons South China Sea maritime disputes Russia Ukraine Iran "
        "Saudi Arabia Baltics Korea Arctic Philippines Africa Caribbean Venezuela NDAA armed forces"
    ),
    "Border Security & Migration": (
        "border enforcement border wall border patrol immigration asylum visas citizenship migration homeland security"
    ),
    "Cross-Border Infrastructure": (
        "ports of entry cross border rail trucking freight corridors border water sanitation environment infrastructure"
    ),
    "USMCA & North American Trade": (
        "USMCA United States Mexico Canada Agreement rules of origin foreign trade zones US Mexico bilateral economic "
        "regional development agricultural trade Mexico North American trade NAFTA"
    ),
    "Political / Foreign Influence": (
        "CCP propaganda disinformation united front influence operations influence transparency disclosure condemnation "
        "Chinese Communist Party political influence foreign influence"
    ),
    "Currency, Financial & Funding": (
        "exchange rates currency manipulation capital markets listings delisting pension index fund exposure payment systems "
        "sovereign debt IMF World Bank IDB finance financial funding securities"
    ),
    "Appropriations Vehicles": (
        "omnibus agency appropriations continuing resolution supplemental appropriations budget funding act spending bill"
    ),
    "State Department & Foreign Aid": (
        "State Department authorization foreign operations appropriations US assistance PRC Countering PRC Influence Fund "
        "embassy diplomatic foreign aid diplomacy secretary of state"
    ),
}

# Strong lexical signals. These are intentionally compact and interpretable.
KEYWORDS: Dict[str, List[str]] = {
    "Forced Labor & Xinjiang": ["forced labor", "forced labour", "xinjiang", "uyghur", "uflpa"],
    "Hong Kong": ["hong kong", "hong kong autonomy"],
    "Taiwan": ["taiwan", "resolution 2758"],
    "Human Rights & Political Repression": ["tibet", "organ harvesting", "religious freedom", "political prisoner", "genocide", "human rights"],
    "COVID-19 & Pandemic Accountability": ["covid", "coronavirus", "pandemic", "wuhan", "world health organization", "who"],
    "Narcotics & Fentanyl": ["fentanyl", "narcotic", "opioid", "drug trafficking", "precursor chemical", "cartel"],
    "Technology Competition, Telecom & Strategic Sectors": ["huawei", "zte", "5g", "semiconductor", "quantum", "artificial intelligence", "biotech", "drone", "undersea cable", "telecom"],
    "Energy, Grid & Electrical Infrastructure": ["bulk-power", "bulk power", "transformer", "substation", "electric grid", "solar", "renewable", "nuclear", "critical mineral", "oil and gas"],
    "Foreign Investment in the US": ["cfius", "farmland", "real estate", "foreign ownership", "foreign investment", "acquisition"],
    "Information & Technology Security": ["trade secret", "intellectual property", "counterfeit", "confucius institute", "talent program", "espionage", "counterintelligence", "cyber", "personal data", "tiktok"],
    "Sanctions & Export Controls": ["sanction", "asset freeze", "entity list", "denial order", "export control", "export license", "dual-use", "ofac", "bis"],
    "Tariffs, Trade Remedies & Market Access": ["tariff", "section 301", "antidumping", "countervailing", "de minimis", "normal trade relations", "ustr", "customs duty", "market access"],
    "Supply Chain Security & Reshoring": ["reshoring", "onshoring", "nearshoring", "supply chain", "industrial base", "stockpile", "domestic manufacturing"],
    "Federal Procurement & Buy American": ["procurement", "buy american", "federal acquisition", "government contract", "domestic content"],
    "Defense, Military & Territorial Disputes": ["national defense authorization", "ndaa", "south china sea", "pla", "military", "armed forces", "weapons", "maritime dispute"],
    "Border Security & Migration": ["border patrol", "border wall", "immigration", "asylum", "visa", "citizenship"],
    "Cross-Border Infrastructure": ["port of entry", "cross-border rail", "freight corridor", "border water", "sanitation"],
    "USMCA & North American Trade": ["usmca", "rules of origin", "nafta", "u.s.-mexico", "us-mexico", "mexico trade"],
    "Political / Foreign Influence": ["propaganda", "disinformation", "united front", "influence operation", "ccp influence", "communist party influence"],
    "Currency, Financial & Funding": ["currency manipulation", "exchange rate", "capital market", "delisting", "pension fund", "index fund", "payment system", "sovereign debt", "imf", "world bank"],
    "Appropriations Vehicles": ["appropriations act", "continuing resolution", "supplemental appropriations", "omnibus appropriations"],
    "State Department & Foreign Aid": ["state department", "foreign operations", "foreign aid", "u.s. assistance", "us assistance", "embassy", "diplomatic", "secretary of state"],
}

# Instrument topics get priority when a bill contains a concrete operative mechanism.
INSTRUMENT_PRIORITY = [
    "Forced Labor & Xinjiang",
    "Hong Kong",
    "Taiwan",
    "COVID-19 & Pandemic Accountability",
    "Narcotics & Fentanyl",
    "Technology Competition, Telecom & Strategic Sectors",
    "Energy, Grid & Electrical Infrastructure",
    "Foreign Investment in the US",
    "Information & Technology Security",
    "Sanctions & Export Controls",
    "Tariffs, Trade Remedies & Market Access",
    "Supply Chain Security & Reshoring",
    "Federal Procurement & Buy American",
    "Defense, Military & Territorial Disputes",
    "Border Security & Migration",
    "Cross-Border Infrastructure",
    "USMCA & North American Trade",
    "Currency, Financial & Funding",
    "Appropriations Vehicles",
    "State Department & Foreign Aid",
    "Human Rights & Political Repression",
    "Political / Foreign Influence",
]


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).lower()
    value = value.replace("–", "-").replace("—", "-")
    value = re.sub(r"[^a-z0-9%$&+./\-\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def keyword_score(text: str, topic: str) -> Tuple[float, List[str]]:
    hits = []
    score = 0.0
    for kw in KEYWORDS.get(topic, []):
        if kw in text:
            hits.append(kw)
            # Multi-word signals are deliberately stronger.
            score += 2.2 if " " in kw else 1.4
    return score, hits


def special_rule_adjustments(text: str, scores: Dict[str, float]) -> Dict[str, float]:
    """Apply the user's decision logic where instrument beats general theme."""
    adjusted = scores.copy()

    # Sanctions/export-control instrument wins over a human-rights rationale.
    if re.search(r"\b(sanction|entity list|asset freeze|export control|export license|denial order)\b", text):
        adjusted["Sanctions & Export Controls"] += 4.0
        adjusted["Human Rights & Political Repression"] -= 1.0
        adjusted["Political / Foreign Influence"] -= 1.0

    # Tariff/customs instrument wins over political framing.
    if re.search(r"\b(tariff|section 301|antidumping|countervailing|de minimis|customs dut)\w*\b", text):
        adjusted["Tariffs, Trade Remedies & Market Access"] += 4.0
        adjusted["Political / Foreign Influence"] -= 1.0

    # Procurement instrument wins over technology/influence rationale.
    if re.search(r"\b(procurement|buy american|federal acquisition|government contract)\b", text):
        adjusted["Federal Procurement & Buy American"] += 4.0
        adjusted["Political / Foreign Influence"] -= 1.0

    # Ownership/acquisition is distinct from money/capital flows.
    if re.search(r"\b(cfius|foreign ownership|acquisition|farmland|real estate)\b", text):
        adjusted["Foreign Investment in the US"] += 4.0

    # State/foreign-operations vehicles get a strong lift.
    if re.search(r"\b(state department|foreign operations|foreign aid|embassy|diplomatic)\b", text):
        adjusted["State Department & Foreign Aid"] += 3.2

    # Appropriations vehicle if the bill itself is an omnibus/CR/supplemental.
    if re.search(r"\b(appropriations act|continuing resolution|supplemental appropriations|omnibus)\b", text):
        adjusted["Appropriations Vehicles"] += 2.8

    # Political / Foreign Influence must not act as a residual bucket.
    concrete_topics = [t for t in INSTRUMENT_PRIORITY if t not in {"Human Rights & Political Repression", "Political / Foreign Influence"}]
    if max(adjusted[t] for t in concrete_topics) > adjusted["Political / Foreign Influence"]:
        adjusted["Political / Foreign Influence"] -= 0.7

    # Human rights only when no concrete economic/trade/sanctions instrument dominates.
    concrete_econ = [
        "Sanctions & Export Controls",
        "Tariffs, Trade Remedies & Market Access",
        "Federal Procurement & Buy American",
        "Foreign Investment in the US",
        "Currency, Financial & Funding",
        "Supply Chain Security & Reshoring",
    ]
    if max(adjusted[t] for t in concrete_econ) > adjusted["Human Rights & Political Repression"]:
        adjusted["Human Rights & Political Repression"] -= 0.8

    return adjusted


def build_classifier(texts: List[str]):
    topic_names = list(TOPICS.keys())
    topic_docs = [f"{name}. {TOPICS[name]}" for name in topic_names]
    corpus = topic_docs + texts
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    topic_matrix = matrix[: len(topic_names)]
    row_matrix = matrix[len(topic_names) :]
    similarities = cosine_similarity(row_matrix, topic_matrix)
    return topic_names, similarities


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Title", "Analytical Summary"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    titles = df["Title"].map(clean_text)
    summaries = df["Analytical Summary"].map(clean_text)

    # Repeat the title to give it additional weight without creating a black-box model.
    texts = [f"{t} {t} {s}".strip() for t, s in zip(titles, summaries)]
    topic_names, similarities = build_classifier(texts)

    assigned = []
    confidences = []
    evidence_list = []
    runner_up_list = []

    for i, text in enumerate(texts):
        scores = {}
        all_hits = {}
        for j, topic in enumerate(topic_names):
            kw_score, hits = keyword_score(text, topic)
            # Hybrid: semantic similarity + transparent keyword evidence.
            scores[topic] = float(similarities[i, j] * 10.0 + kw_score)
            all_hits[topic] = hits

        scores = special_rule_adjustments(text, scores)
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        winner, top_score = ordered[0]
        runner_up, second_score = ordered[1]

        # Margin-based confidence. It is a ranking aid, not a probability.
        margin = max(0.0, top_score - second_score)
        confidence = min(99.0, 52.0 + (margin * 7.0) + min(top_score, 5.0) * 4.0)

        hits = all_hits.get(winner, [])
        title_raw = "" if pd.isna(df.iloc[i]["Title"]) else str(df.iloc[i]["Title"])
        summary_raw = "" if pd.isna(df.iloc[i]["Analytical Summary"]) else str(df.iloc[i]["Analytical Summary"])
        excerpt = summary_raw[:320].strip()
        if len(summary_raw) > 320:
            excerpt += "…"

        if hits:
            evidence = f"Matched: {', '.join(hits[:8])}. Title: {title_raw}. Summary evidence: {excerpt}"
        else:
            evidence = f"Semantic match from Title + Analytical Summary. Title: {title_raw}. Summary evidence: {excerpt}"

        assigned.append(winner)
        confidences.append(round(confidence, 1))
        evidence_list.append(evidence)
        runner_up_list.append(runner_up)

    out = df.copy()
    out.insert(0, "Topic", assigned)
    out["Topic Confidence"] = confidences
    out["Topic Runner-Up"] = runner_up_list
    out["Topic Evidence"] = evidence_list
    return out


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Classified Bills")
        workbook = writer.book
        worksheet = writer.sheets["Classified Bills"]

        header_fmt = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": "#0F4C5C",
            "border": 0,
            "align": "center",
            "valign": "vcenter",
        })
        wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
        pct_fmt = workbook.add_format({"num_format": "0.0", "valign": "top"})

        for col_idx, col_name in enumerate(df.columns):
            worksheet.write(0, col_idx, col_name, header_fmt)
            if col_name in {"Title", "Analytical Summary", "Topic Evidence"}:
                worksheet.set_column(col_idx, col_idx, 42, wrap_fmt)
            elif col_name == "Topic":
                worksheet.set_column(col_idx, col_idx, 38, wrap_fmt)
            elif col_name == "Topic Confidence":
                worksheet.set_column(col_idx, col_idx, 16, pct_fmt)
            else:
                worksheet.set_column(col_idx, col_idx, 18)

        worksheet.freeze_panes(1, 1)
        worksheet.autofilter(0, 0, len(df), len(df.columns) - 1)
        worksheet.set_row(0, 24)

    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# STREAMLIT UI
# ============================================================
# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(
    page_title="Congress Bill Topic Intelligence",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- visual polish ----------
st.markdown("""
<style>
.block-container {padding-top: 1.7rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255,255,255,.96), rgba(245,247,250,.96));
    border: 1px solid rgba(49,51,63,.10);
    padding: 14px 16px;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,.04);
}
.evidence-card {
    border-radius: 18px;
    padding: 18px 20px;
    margin: 8px 0 14px 0;
    border-left: 7px solid;
    box-shadow: 0 5px 18px rgba(0,0,0,.06);
}
.ev-blue   {background:#EAF3FF; border-color:#2F80ED;}
.ev-green  {background:#EAFBF1; border-color:#27AE60;}
.ev-purple {background:#F3ECFF; border-color:#8E44AD;}
.ev-orange {background:#FFF4E5; border-color:#F2994A;}
.ev-red    {background:#FFECEC; border-color:#EB5757;}
.ev-title {font-size:.78rem; font-weight:800; letter-spacing:.06em; text-transform:uppercase; opacity:.72;}
.ev-big {font-size:1.05rem; font-weight:750; margin-top:4px;}
.ev-body {font-size:.94rem; line-height:1.45; margin-top:5px;}
.hero-note {
    background: linear-gradient(90deg,#EEF6FF,#F7F1FF);
    border:1px solid #D9E7FF; border-radius:18px; padding:14px 18px; margin-bottom:14px;
}
</style>
""", unsafe_allow_html=True)

def esc(value) -> str:
    import html
    return html.escape("" if pd.isna(value) else str(value))

def evidence_card(label: str, body: str, css_class: str, big: bool = False):
    body_class = "ev-big" if big else "ev-body"
    st.markdown(
        f'<div class="evidence-card {css_class}">'
        f'<div class="ev-title">{esc(label)}</div>'
        f'<div class="{body_class}">{esc(body)}</div></div>',
        unsafe_allow_html=True,
    )

st.title("🧭 Congress Bill Topic Intelligence")
st.markdown(
    '<div class="hero-note"><b>Upload → classify → inspect evidence → explore patterns → export.</b> '
    'Each bill is assigned to exactly one of the 22 approved topics using its '
    '<b>Title + Analytical Summary</b>.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("📂 Data")
    uploaded = st.file_uploader(
        "Upload the master Excel file",
        type=["xlsx", "xls"],
        help="The workbook must contain Title and Analytical Summary columns.",
    )
    st.caption("Nothing is read from a local default path — you choose the file.")
    st.divider()
    st.markdown("**🧠 Classifier**")
    st.write("TF-IDF similarity + high-signal topic terms + instrument-priority rules.")
    st.write("No OpenAI API key is required.")
    st.divider()
    st.markdown("**Required columns**")
    st.code("Title\nAnalytical Summary", language=None)

if uploaded is None:
    st.info("👈 Upload your master Excel file from the sidebar to start.")
    st.stop()

try:
    raw_df = pd.read_excel(uploaded)
except Exception as exc:
    st.error(f"Could not read the Excel file: {exc}")
    st.stop()

st.success(f"✅ Loaded {len(raw_df):,} rows and {len(raw_df.columns):,} columns.")

with st.spinner("Classifying bills into the 22 topics…"):
    try:
        result_df = classify_dataframe(raw_df)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

# ---------------- KPIs ----------------
low_cutoff = 65
top_topic = result_df["Topic"].value_counts().idxmax()
top_topic_n = int(result_df["Topic"].value_counts().max())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Bills classified", f"{len(result_df):,}")
c2.metric("Topics represented", f"{result_df['Topic'].nunique()}/22")
c3.metric("Avg. confidence", f"{result_df['Topic Confidence'].mean():.1f}/100")
c4.metric("Needs review", f"{(result_df['Topic Confidence'] < low_cutoff).sum():,}")
c5.metric("Largest topic", f"{top_topic_n:,}", help=top_topic)

st.divider()

counts = result_df["Topic"].value_counts().rename_axis("Topic").reset_index(name="Bills")
counts["Share"] = counts["Bills"] / counts["Bills"].sum()

# ---------------- CHARTS ----------------
st.header("📊 Topic landscape")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Easy-read ranking", "Share & composition", "Confidence", "Congress trends"]
)

with tab1:
    fig_bar = px.bar(
        counts.sort_values("Bills", ascending=True),
        x="Bills", y="Topic", orientation="h", text="Bills",
        color="Bills", color_continuous_scale="Blues",
        title="How many bills belong to each topic?",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        height=max(560, 30 * len(counts)),
        yaxis_title="", xaxis_title="Number of bills",
        coloraxis_showscale=False, margin=dict(l=10, r=35, t=55, b=30),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    p1, p2 = st.columns(2)
    with p1:
        fig_donut = px.pie(
            counts, names="Topic", values="Bills", hole=.55,
            title="Topic share",
        )
        fig_donut.update_traces(textposition="inside", textinfo="percent")
        fig_donut.update_layout(height=520, legend=dict(orientation="h", y=-.2))
        st.plotly_chart(fig_donut, use_container_width=True)
    with p2:
        fig_tree = px.treemap(
            counts, path=["Topic"], values="Bills", color="Bills",
            color_continuous_scale="Viridis",
            title="Topic map — larger box = more bills",
        )
        fig_tree.update_layout(height=520, coloraxis_showscale=False)
        st.plotly_chart(fig_tree, use_container_width=True)

with tab3:
    q1, q2 = st.columns(2)
    with q1:
        fig_hist = px.histogram(
            result_df, x="Topic Confidence", nbins=20,
            color_discrete_sequence=["#2F80ED"],
            title="Confidence distribution",
        )
        fig_hist.add_vline(x=low_cutoff, line_dash="dash", annotation_text="review line")
        fig_hist.update_layout(height=440, yaxis_title="Bills")
        st.plotly_chart(fig_hist, use_container_width=True)
    with q2:
        conf_topic = (
            result_df.groupby("Topic", as_index=False)["Topic Confidence"]
            .mean().sort_values("Topic Confidence")
        )
        fig_conf = px.bar(
            conf_topic, x="Topic Confidence", y="Topic", orientation="h",
            color="Topic Confidence", color_continuous_scale="Tealgrn",
            title="Average confidence by topic",
        )
        fig_conf.update_layout(
            height=max(440, 27 * len(conf_topic)), yaxis_title="",
            coloraxis_showscale=False, xaxis_range=[0, 100],
        )
        st.plotly_chart(fig_conf, use_container_width=True)

with tab4:
    if "Congress" in result_df.columns:
        cross = result_df.groupby(["Congress", "Topic"], dropna=False).size().reset_index(name="Bills")
        chart_kind = st.radio(
            "View",
            ["Stacked bars", "Heatmap"],
            horizontal=True,
            key="congress_chart_kind",
        )
        if chart_kind == "Stacked bars":
            fig_cross = px.bar(
                cross, x="Congress", y="Bills", color="Topic",
                title="Topic mix by Congress",
            )
            fig_cross.update_layout(height=540, legend_title_text="Topic")
            st.plotly_chart(fig_cross, use_container_width=True)
        else:
            pivot = cross.pivot(index="Topic", columns="Congress", values="Bills").fillna(0)
            fig_heat = px.imshow(
                pivot, aspect="auto", text_auto=True,
                color_continuous_scale="Blues",
                title="Congress × Topic heatmap",
            )
            fig_heat.update_layout(height=max(520, 30 * len(pivot)), xaxis_title="Congress", yaxis_title="")
            st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("A Congress column was not found, so this view is hidden.")

st.divider()

# ---------------- EVIDENCE EXPLORER ----------------
st.header("🔎 Evidence Explorer")
f1, f2, f3 = st.columns([1.6, 1, 1])
with f1:
    search_text = st.text_input(
        "Search",
        placeholder="fentanyl, Taiwan, export controls, bill title…",
    )
with f2:
    topic_filter = st.selectbox("Topic", ["All topics"] + sorted(result_df["Topic"].unique().tolist()))
with f3:
    review_only = st.toggle(f"Only confidence < {low_cutoff}")

view_df = result_df.copy()
if search_text.strip():
    needle = re.escape(search_text.strip())
    mask = (
        view_df["Title"].astype(str).str.contains(needle, case=False, na=False, regex=True)
        | view_df["Analytical Summary"].astype(str).str.contains(needle, case=False, na=False, regex=True)
        | view_df["Topic"].astype(str).str.contains(needle, case=False, na=False, regex=True)
    )
    view_df = view_df.loc[mask]
if topic_filter != "All topics":
    view_df = view_df.loc[view_df["Topic"] == topic_filter]
if review_only:
    view_df = view_df.loc[view_df["Topic Confidence"] < low_cutoff]

st.caption(f"Showing {len(view_df):,} matching row(s).")

if len(view_df):
    label_options = {
        idx: f"{row.get('Bill', idx)} | {row['Title']} → {row['Topic']}"
        for idx, row in view_df.head(1000).iterrows()
    }
    selected_idx = st.selectbox(
        "Choose a bill to inspect",
        options=list(label_options.keys()),
        format_func=lambda idx: label_options[idx],
    )
    row = result_df.loc[selected_idx]

    main, evidence = st.columns([1.35, 1], gap="large")

    with main:
        st.subheader("Bill text")
        evidence_card("Title", row["Title"], "ev-blue", big=True)
        evidence_card("Analytical Summary", row["Analytical Summary"], "ev-purple")
        if "Mechanism" in row.index and pd.notna(row["Mechanism"]):
            evidence_card("Mechanism", row["Mechanism"], "ev-orange")

    with evidence:
        st.subheader("Classification evidence")
        confidence = float(row["Topic Confidence"])
        topic_css = "ev-green" if confidence >= 75 else ("ev-orange" if confidence >= low_cutoff else "ev-red")
        evidence_card("Assigned topic", row["Topic"], topic_css, big=True)
        st.progress(min(confidence / 100.0, 1.0))
        evidence_card("Confidence", f"{confidence:.1f}/100", topic_css)
        evidence_card("Runner-up topic", row["Topic Runner-Up"], "ev-purple")
        evidence_card("Why the model chose it", row["Topic Evidence"], "ev-blue")
else:
    st.warning("No rows match the current filters.")

st.divider()

# ---------------- REVIEW QUEUE ----------------
st.header("🧪 Quick review queue")
st.caption("Borderline rows appear first so a human reviewer can validate the hardest cases quickly.")
review_cols = [
    c for c in ["Topic", "Topic Confidence", "Topic Runner-Up", "Bill", "Title", "Analytical Summary"]
    if c in result_df.columns
]
review_df = result_df.sort_values("Topic Confidence", ascending=True)[review_cols].head(50)
st.dataframe(
    review_df,
    use_container_width=True,
    height=360,
    hide_index=True,
    column_config={
        "Topic Confidence": st.column_config.ProgressColumn(
            "Confidence", min_value=0, max_value=100, format="%.1f"
        )
    },
)

st.divider()

# ---------------- DATA + EXPORT ----------------
st.header("📦 Processed data")
show_cols = [
    c for c in ["Topic", "Congress", "Bill", "Title", "Analytical Summary", "Topic Confidence", "Topic Runner-Up"]
    if c in result_df.columns
]
st.dataframe(result_df[show_cols], use_container_width=True, height=440, hide_index=True)

excel_bytes = to_excel_bytes(result_df)
st.download_button(
    "⬇️ Download post-processed Excel",
    data=excel_bytes,
    file_name="Congress_Bill_Gate_2_Classified.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
    type="primary",
)

st.caption(
    "The exported workbook keeps the original data, inserts Topic as the first column, "
    "and adds confidence, runner-up and evidence fields for auditability."
)
