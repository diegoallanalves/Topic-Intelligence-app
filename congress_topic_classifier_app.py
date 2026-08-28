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
# CONFIG — PROVISION-FIRST, NO AI / NO API
# ============================================================
TOPICS: Dict[str, str] = {
    "Forced Labor & Xinjiang": "forced-labour import bans rebuttable presumptions Xinjiang entity lists supply-chain forced-labour disclosure",
    "Hong Kong": "Hong Kong autonomy certification separate customs export-control status sanctions officials",
    "Taiwan": "Taiwan relations arms sales international participation trade investment agreements UNGA Resolution 2758",
    "Human Rights & Political Repression": "Tibet organ harvesting religious freedom genocide political prisoners censorship Congressional Gold Medals transnational repression",
    "COVID-19 & Pandemic Accountability": "COVID pandemic origins WHO liability PPE medical-supply dependence",
    "Narcotics & Fentanyl": "fentanyl precursor scheduling trafficking cartels drug-related money laundering",
    "Technology Competition, Telecom & Strategic Sectors": "Huawei ZTE 5G telecom network equipment undersea cables equipment authorisation semiconductors quantum AI biotech drones aerospace technical standards",
    "Energy, Grid & Electrical Infrastructure": "bulk-power system transformers substations switchgear transmission generation solar photovoltaic renewables nuclear oil gas Strategic Petroleum Reserve critical minerals metals",
    "Foreign Investment in the US": "CFIUS ownership acquisitions farmland real estate strategic assets joint ventures foreign-owned firms",
    "Information & Technology Security": "IP trade-secret theft counterfeiting academic research security Confucius Institutes talent programmes espionage counterintelligence foreign agents cyber personal data apps platforms PRC visa screening researchers students military-affiliated applicants",
    "Sanctions & Export Controls": "entity designations asset freezes entity lists denial orders export licensing dual-use controls",
    "Tariffs, Trade Remedies & Market Access": "tariffs duties customs de minimis Section 301 antidumping countervailing duties normal trade relations market-economy developing-country status unfair trade subsidies USTR",
    "Supply Chain Security & Reshoring": "critical supply-chain reviews reshoring onshoring nearshoring incentives relocation incentives domestic manufacturing industrial base stockpiles",
    "Federal Procurement & Buy American": "government contractor procurement prohibitions Buy American domestic-content federal acquisition public works",
    "Defense, Military & Territorial Disputes": "defence authorisation posture PLA weapons South China Sea maritime disputes Russia Ukraine Iran Saudi Baltics Korea Arctic Philippines Africa Caribbean Venezuela",
    "Border Security & Migration": "US-Mexico border enforcement wall border patrol ports-of-entry staffing asylum migration surges",
    "Cross-Border Infrastructure": "ports of entry cross-border rail trucking freight corridors border water sanitation environment",
    "USMCA & North American Trade": "USMCA rules of origin foreign-trade zones US-Mexico economic partnership regional development agricultural trade Mexico",
    "Political / Foreign Influence": "CCP propaganda disinformation united-front influence operations transparency disclosure general condemnation",
    "Currency, Financial & Funding": "exchange rates currency manipulation capital markets listings delisting pension index funds payment systems sovereign debt IMF World Bank IDB",
    "Appropriations Vehicles": "omnibus agency appropriations continuing resolutions supplementals",
    "State Department & Foreign Aid": "State Department authorisation restrictions US assistance PRC Countering PRC Influence Fund embassy diplomatic provisions",
}

KEYWORDS: Dict[str, List[str]] = {
    "Forced Labor & Xinjiang": ["forced labor", "forced labour", "rebuttable presumption", "uflpa", "xinjiang entity", "uyghur forced"],
    "Hong Kong": ["hong kong autonomy", "hong kong customs", "hong kong"],
    "Taiwan": ["taiwan", "resolution 2758", "unga 2758"],
    "Human Rights & Political Repression": ["tibet", "organ harvesting", "religious freedom", "political prisoner", "genocide", "human rights", "gold medal", "transnational repression"],
    "COVID-19 & Pandemic Accountability": ["covid", "coronavirus", "pandemic", "wuhan", "world health organization", "ppe"],
    "Narcotics & Fentanyl": ["fentanyl", "narcotic", "opioid", "drug trafficking", "precursor chemical", "cartel"],
    "Technology Competition, Telecom & Strategic Sectors": ["huawei", "zte", "5g", "telecom", "network equipment", "undersea cable", "semiconductor", "quantum", "artificial intelligence", "biotech", "drone", "aerospace", "technical standard"],
    "Energy, Grid & Electrical Infrastructure": ["bulk-power", "bulk power", "transformer", "substation", "switchgear", "transmission", "generation", "solar", "photovoltaic", "renewable", "nuclear", "strategic petroleum reserve", "critical mineral", "oil", "gas"],
    "Foreign Investment in the US": ["cfius", "foreign ownership", "acquire", "acquisition", "farmland", "real estate", "joint venture"],
    "Information & Technology Security": ["trade secret", "intellectual property", "counterfeit", "confucius institute", "talent program", "espionage", "counterintelligence", "foreign agent", "cyber", "personal data", "tiktok", "visa"],
    "Sanctions & Export Controls": ["sanction", "asset freeze", "entity list", "denial order", "export control", "export license", "export licensing", "dual-use", "ofac", "bis"],
    "Tariffs, Trade Remedies & Market Access": ["tariff", "section 301", "antidumping", "countervailing", "de minimis", "normal trade relations", "market economy", "nonmarket economy", "developing-country", "ustr", "customs duty", "trade remedy"],
    "Supply Chain Security & Reshoring": ["reshoring", "onshoring", "nearshoring", "critical supply chain", "industrial base", "stockpile", "domestic manufacturing", "relocation incentive"],
    "Federal Procurement & Buy American": ["procurement", "buy american", "federal acquisition", "government contract", "domestic content", "purchasing"],
    "Defense, Military & Territorial Disputes": ["national defense authorization", "ndaa", "south china sea", "pla", "military", "armed forces", "weapons", "maritime dispute"],
    "Border Security & Migration": ["southern border", "u.s.-mexico border", "us-mexico border", "border patrol", "border wall", "asylum", "migration"],
    "Cross-Border Infrastructure": ["port of entry", "ports of entry", "cross-border rail", "freight corridor", "border water", "sanitation"],
    "USMCA & North American Trade": ["usmca", "rules of origin", "foreign-trade zone", "nafta", "u.s.-mexico", "us-mexico", "agricultural trade"],
    "Political / Foreign Influence": ["propaganda", "disinformation", "united front", "influence operation", "influence transparency", "condemnation"],
    "Currency, Financial & Funding": ["currency manipulation", "exchange rate", "capital market", "listing", "delisting", "pension fund", "index fund", "payment system", "sovereign debt", "imf", "world bank", "idb"],
    "Appropriations Vehicles": ["appropriations act", "continuing resolution", "continuing appropriations", "making appropriations", "supplemental appropriations"],
    "State Department & Foreign Aid": ["state department", "foreign aid", "u.s. assistance", "us assistance", "countering prc influence fund", "embassy", "diplomatic"],
}

LEGAL_EFFECT = [
    "prohibit", "bar", "require", "impose", "levy", "authoriz", "authoris",
    "appropriate", "fund", "establish", "direct", "restrict", "suspend",
    "withdraw", "designate", "amend", "condition", "screen", "certif",
    "repeal", "waive", "allocate", "maintain"
]
CONSEQUENCE_MARKERS = [
    "could", "may", "might", "potentially", "likely", "reflects", "signals",
    "sentiment", "risk", "daqO", "poses a", "for daqo", "daqo's"
]
NO_SUBJECT_WORDS = [
    "america", "american", "asia", "security", "infrastructure", "innovation",
    "technology", "science", "leadership", "threat", "foreign", "strategic",
    "competition", "ccp", "china", "influence", "data", "space", "ports"
]
APPROPRIATIONS_TITLE = [
    "appropriations act", "continuing resolution", "continuing appropriations",
    "making appropriations", "supplemental appropriations"
]
ENERGY_TERMS = [
    "grid", "bulk-power", "bulk power", "transmission", "substation", "transformer",
    "switchgear", "generation", "solar", "photovoltaic", "nuclear", "oil", "gas",
    "strategic petroleum reserve", "critical mineral"
]
THIRD_COUNTRIES = [
    "russia", "ukraine", "iran", "saudi", "baltic", "korea", "arctic",
    "philippines", "africa", "caribbean", "venezuela"
]


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).lower().replace("–", "-").replace("—", "-")
    value = re.sub(r"[^a-z0-9%$&+./'\-\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def split_sentences(value) -> List[str]:
    raw = "" if pd.isna(value) else str(value)
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", raw) if s.strip()]


def is_consequence(sentence: str) -> bool:
    s = clean_text(sentence)
    if "daqO".lower() in s:
        return True
    return any(re.search(rf"\b{re.escape(clean_text(m))}\b", s) for m in CONSEQUENCE_MARKERS if clean_text(m))


def has_legal_effect(sentence: str) -> bool:
    s = clean_text(sentence)
    return any(re.search(rf"\b\w*{re.escape(stem)}\w*\b", s) for stem in LEGAL_EFFECT)


def extract_provision(row: pd.Series) -> Tuple[str, str]:
    """Mechanism first, Summary second, Title only as fallback."""
    for source in ["Mechanism", "Analytical Summary"]:
        if source not in row.index:
            continue
        for sentence in split_sentences(row.get(source, "")):
            if not is_consequence(sentence) and has_legal_effect(sentence):
                return sentence.strip(), source

    # No stated provision in the two substantive fields.
    title = "" if pd.isna(row.get("Title", "")) else str(row.get("Title", "")).strip()
    if title and has_legal_effect(title):
        return title, "Title"
    return "NONE STATED", "None"


def keyword_score(text: str, topic: str) -> Tuple[float, List[str]]:
    hits, score = [], 0.0
    for kw in KEYWORDS.get(topic, []):
        if kw in text:
            hits.append(kw)
            score += 2.2 if " " in kw else 1.4
    return score, hits


def build_classifier(texts: List[str]):
    topic_names = list(TOPICS.keys())
    topic_docs = [f"{name}. {TOPICS[name]}" for name in topic_names]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1, sublinear_tf=True)
    matrix = vectorizer.fit_transform(topic_docs + texts)
    similarities = cosine_similarity(matrix[len(topic_names):], matrix[:len(topic_names)])
    return topic_names, similarities


def title_rule(title: str, provision: str) -> Tuple[str | None, str | None, str | None]:
    """Strict ladder rules that legitimately use Title."""
    t = clean_text(title)
    p = clean_text(provision)

    # Rule 1
    if any(x in t for x in APPROPRIATIONS_TITLE):
        return "Appropriations Vehicles", "State Department & Foreign Aid", "rule 1"

    # Rule 2a
    if "taiwan" in t:
        return "Taiwan", "Sanctions & Export Controls", "rule 2a"
    if "hong kong" in t:
        return "Hong Kong", "Sanctions & Export Controls", "rule 2a"
    if "uyghur" in t or "xinjiang" in t:
        trade = any(x in p for x in ["import ban", "entity list", "rebuttable presumption", "supply-chain", "supply chain", "forced labor", "forced labour"])
        if trade:
            return "Forced Labor & Xinjiang", "Human Rights & Political Repression", "rule 2a"
        return "Human Rights & Political Repression", "Forced Labor & Xinjiang", "rule 2a"

    return None, None, None


def provision_rule(provision: str) -> Tuple[str | None, str | None, str | None]:
    p = clean_text(provision)

    # Rule 2b: energy/electrical thing beats procurement.
    if any(x in p for x in ENERGY_TERMS):
        runner = "Federal Procurement & Buy American" if any(x in p for x in ["procure", "purchas", "government", "contractor"]) else None
        return "Energy, Grid & Electrical Infrastructure", runner, "rule 2b"

    # Rule 3: government procurement, non-energy.
    if any(x in p for x in ["procure", "purchas", "buy american", "federal acquisition", "government contract", "contractor may buy"]):
        return "Federal Procurement & Buy American", None, "rule 3"

    # Rule 4: ownership/acquisition vs money/capital.
    if any(x in p for x in ["cfius", "own", "ownership", "acquir", "farmland", "real estate", "joint venture"]):
        return "Foreign Investment in the US", "Currency, Financial & Funding", "rule 4"
    if any(x in p for x in ["currency", "exchange rate", "capital market", "listing", "delisting", "pension", "index fund", "payment system", "sovereign debt", "imf", "world bank", "idb"]):
        return "Currency, Financial & Funding", "Foreign Investment in the US", "rule 4"

    # Rule 5: visa split.
    if "visa" in p or "applicant" in p:
        if any(x in p for x in ["prc", "chinese", "researcher", "student", "military-affiliated", "military affiliated"]):
            return "Information & Technology Security", "Border Security & Migration", "rule 5"
        if any(x in p for x in ["southern border", "mexico", "asylum", "border patrol", "migration"]):
            return "Border Security & Migration", "Information & Technology Security", "rule 5"

    # Rule 6: third country, unless operative Chinese sanctions/tariff/export control.
    if any(x in p for x in THIRD_COUNTRIES):
        if any(x in p for x in ["tariff", "trade penalt", "duty", "section 301"]):
            return "Tariffs, Trade Remedies & Market Access", "Defense, Military & Territorial Disputes", "rule 6"
        if any(x in p for x in ["sanction", "asset freeze", "entity list", "export control", "export licens"]):
            return "Sanctions & Export Controls", "Defense, Military & Territorial Disputes", "rule 6"
        return "Defense, Military & Territorial Disputes", None, "rule 6"

    return None, None, None


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Title", "Analytical Summary", "Mechanism"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    provisions, sources = [], []
    for _, row in df.iterrows():
        provision, source = extract_provision(row)
        provisions.append(provision)
        sources.append(source)

    semantic_texts = [clean_text(p) if p != "NONE STATED" else "" for p in provisions]
    topic_names, similarities = build_classifier(semantic_texts)

    assigned, confidences, runners, evidence = [], [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        title = str(row.get("Title", "") or "")
        provision = provisions[i]
        source = sources[i]

        # Title is consulted only for the allowed title rules.
        winner, runner, rule = title_rule(title, provision)

        # Then apply provision ladder.
        if winner is None and provision != "NONE STATED":
            winner, runner, rule = provision_rule(provision)

        scores, hits_by_topic = {}, {}
        ptext = clean_text(provision)

        # If no strict tie-break decided the row, classify ONLY the provision.
        if winner is None and provision != "NONE STATED":
            for j, topic in enumerate(topic_names):
                ks, hits = keyword_score(ptext, topic)
                scores[topic] = float(similarities[i, j] * 10.0 + ks)
                hits_by_topic[topic] = hits
            ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            winner = ordered[0][0]
            runner = ordered[1][0]
            rule = None

        # NONE STATED: Title fallback only. Never classify on consequence text.
        if provision == "NONE STATED":
            t = clean_text(title)
            title_winner, title_runner, title_rule_name = title_rule(title, provision)
            if title_winner:
                winner, runner, rule = title_winner, title_runner, title_rule_name
                confidence = 70
            else:
                # Title-only lexical fallback. If it contains no usable subject,
                # use the mandated last-resort topic and score 50.
                title_scores = {}
                for topic in topic_names:
                    ks, _ = keyword_score(t, topic)
                    title_scores[topic] = ks
                ordered = sorted(title_scores.items(), key=lambda x: x[1], reverse=True)
                if ordered and ordered[0][1] > 0:
                    winner = ordered[0][0]
                    runner = ordered[1][0] if len(ordered) > 1 else "Political / Foreign Influence"
                    confidence = 70
                else:
                    winner = "Political / Foreign Influence"
                    runner = "Human Rights & Political Repression"
                    confidence = 50
        else:
            # Rubric confidence: 85 when a numbered tie-break decides; otherwise 95.
            confidence = 85 if rule else 95

        if not runner or runner == winner:
            # Derive a genuine second-best from the provision only.
            alt_scores = {}
            for j, topic in enumerate(topic_names):
                if topic == winner:
                    continue
                ks, _ = keyword_score(ptext, topic)
                alt_scores[topic] = float(similarities[i, j] * 10.0 + ks)
            runner = max(alt_scores, key=alt_scores.get) if alt_scores else "Political / Foreign Influence"

        runner_display = f"{runner} ({rule})" if rule and provision != "NONE STATED" else runner

        if provision == "NONE STATED":
            why = f"PROVISION: NONE STATED. Classified from Title fallback only. Source: {source}."
        else:
            why = f"Extracted from {source}. Classified on PROVISION only: {provision}"
            if rule:
                why += f" Strict tie-break {rule} decided the row."

        assigned.append(winner)
        confidences.append(confidence)
        runners.append(runner_display)
        evidence.append(why)

    out = df.copy()
    out.insert(0, "Topic", assigned)
    out.insert(1, "Provision", provisions)
    out["Provision Source"] = sources
    out["Topic Confidence"] = confidences
    out["Topic Runner-Up"] = runners
    out["Topic Evidence"] = evidence
    return out


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Classified Bills")
        review = df[df["Topic Confidence"].isin([50, 70])]
        review.to_excel(writer, index=False, sheet_name="Human Review")

        workbook = writer.book
        header_fmt = workbook.add_format({
            "bold": True, "font_color": "white", "bg_color": "#0F4C5C",
            "align": "center", "valign": "vcenter"
        })
        wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})

        for sheet_name, sheet_df in [("Classified Bills", df), ("Human Review", review)]:
            ws = writer.sheets[sheet_name]
            for col_idx, col_name in enumerate(sheet_df.columns):
                ws.write(0, col_idx, col_name, header_fmt)
                if col_name in {"Title", "Mechanism", "Analytical Summary", "Provision", "Topic Evidence"}:
                    ws.set_column(col_idx, col_idx, 42, wrap_fmt)
                elif col_name in {"Topic", "Topic Runner-Up"}:
                    ws.set_column(col_idx, col_idx, 38, wrap_fmt)
                else:
                    ws.set_column(col_idx, col_idx, 18)
            ws.freeze_panes(1, 1)
            ws.autofilter(0, 0, max(len(sheet_df), 1), len(sheet_df.columns) - 1)
            ws.set_row(0, 24)

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
    'Each bill is assigned to exactly one of the 22 approved topics using a '
    '<b>Mechanism → Analytical Summary → Title fallback</b> provision-first method.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("📂 Data")
    uploaded = st.file_uploader(
        "Upload the master Excel file",
        type=["xlsx", "xls"],
        help="The workbook must contain Mechanism, Analytical Summary and Title columns.",
    )
    st.caption("Nothing is read from a local default path — you choose the file.")
    st.divider()
    st.markdown("**🧠 Classifier**")
    st.write("Provision-first rules + TF-IDF/keywords applied only to the extracted provision.")
    st.write("No OpenAI API key is required.")
    st.divider()
    st.markdown("**Required columns**")
    st.code("Mechanism\nAnalytical Summary\nTitle", language=None)

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
low_cutoff = 85
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
    review_only = st.toggle("Only human-review scores (50 / 70)")

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
    view_df = view_df.loc[view_df["Topic Confidence"].isin([50, 70])]

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
        if "Mechanism" in row.index and pd.notna(row["Mechanism"]):
            evidence_card("Mechanism — first source", row["Mechanism"], "ev-orange", big=True)
        evidence_card("Analytical Summary — gap fill / confirmation", row["Analytical Summary"], "ev-purple")
        evidence_card("Title — tie-break / fallback only", row["Title"], "ev-blue")

    with evidence:
        st.subheader("Classification evidence")
        confidence = float(row["Topic Confidence"])
        topic_css = "ev-green" if confidence >= 75 else ("ev-orange" if confidence >= low_cutoff else "ev-red")
        evidence_card("Extracted provision", row["Provision"], "ev-green", big=True)
        evidence_card("Assigned topic", row["Topic"], topic_css, big=True)
        st.progress(min(confidence / 100.0, 1.0))
        evidence_card("Confidence", f"{int(confidence)}/100", topic_css)
        evidence_card("Runner-up topic", row["Topic Runner-Up"], "ev-purple")
        evidence_card("Why the model chose it", row["Topic Evidence"], "ev-blue")
else:
    st.warning("No rows match the current filters.")

st.divider()

# ---------------- REVIEW QUEUE ----------------
st.header("🧪 Quick review queue")
st.caption("Borderline rows appear first so a human reviewer can validate the hardest cases quickly.")
review_cols = [
    c for c in ["Topic", "Provision", "Provision Source", "Topic Confidence", "Topic Runner-Up", "Bill", "Title", "Mechanism", "Analytical Summary"]
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
            "Confidence", min_value=0, max_value=100, format="%d"
        )
    },
)

st.divider()

# ---------------- DATA + EXPORT ----------------
st.header("📦 Processed data")
show_cols = [
    c for c in ["Topic", "Provision", "Provision Source", "Congress", "Bill", "Title", "Mechanism", "Analytical Summary", "Topic Confidence", "Topic Runner-Up"]
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
    "and adds the extracted Provision, its source, rubric confidence, runner-up and evidence fields for auditability."
)
