"""Streamlit theme injection and HTML micro-components."""

from __future__ import annotations

import html
import streamlit as st


def inject_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

:root {
  --bg: oklch(0.15 0.012 52);
  --surface: oklch(0.20 0.014 50);
  --surface-raised: oklch(0.24 0.016 48);
  --text: oklch(0.93 0.018 88);
  --muted: oklch(0.70 0.022 72);
  --accent: oklch(0.70 0.13 42);
  --accent-soft: oklch(0.70 0.13 42 / 0.14);
  --sage: oklch(0.74 0.07 148);
  --sage-soft: oklch(0.74 0.07 148 / 0.14);
  --amber: oklch(0.82 0.11 85);
  --border: oklch(0.34 0.018 52);
  --radius: 14px;
}

.stApp {
  background:
    radial-gradient(ellipse 80% 50% at 100% -10%, oklch(0.28 0.06 42 / 0.35), transparent 55%),
    radial-gradient(ellipse 60% 40% at 0% 100%, oklch(0.22 0.04 148 / 0.2), transparent 50%),
    var(--bg);
  color: var(--text);
  font-family: "Archivo", sans-serif;
}

.block-container {
  padding-top: 1.25rem;
  max-width: 1180px;
}

h1, h2, h3, h4, h5, h6, [data-testid="stMarkdownContainer"] h1 {
  font-family: "Archivo", sans-serif !important;
  letter-spacing: -0.02em;
}

[data-testid="stSidebar"] {
  background: oklch(0.17 0.013 52);
  border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label {
  color: var(--muted);
}

.stTabs [data-baseweb="tab-list"] {
  gap: 6px;
  background: transparent;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0;
}

.stTabs [data-baseweb="tab"] {
  background: transparent;
  border-radius: 10px 10px 0 0;
  color: var(--muted);
  font-weight: 600;
  padding: 10px 18px;
  border: none;
}

.stTabs [aria-selected="true"] {
  background: var(--surface-raised) !important;
  color: var(--text) !important;
}

div[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 16px;
}

div[data-testid="stMetric"] label {
  color: var(--muted) !important;
  font-size: 0.72rem !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: var(--text) !important;
  font-weight: 700;
}

.stButton > button[kind="primary"] {
  background: var(--accent);
  color: oklch(0.18 0.02 52);
  border: none;
  border-radius: 999px;
  font-weight: 700;
  padding: 0.55rem 1.4rem;
}

.stButton > button[kind="primary"]:hover {
  background: oklch(0.76 0.14 42);
  color: oklch(0.18 0.02 52);
}

.stButton > button[kind="secondary"] {
  background: var(--surface-raised);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 999px;
}

[data-testid="stExpander"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}

.stSelectbox div[data-baseweb="select"] > div,
.stTextInput input {
  background: var(--surface) !important;
  border-color: var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
}

.bct-header {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 28px;
  padding-bottom: 22px;
  border-bottom: 1px solid var(--border);
}

.bct-kicker {
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent);
  font-weight: 700;
  margin: 0 0 6px 0;
}

.bct-title {
  font-size: clamp(1.8rem, 3vw, 2.45rem);
  line-height: 1.05;
  font-weight: 700;
  margin: 0;
  color: var(--text);
}

.bct-subtitle {
  margin: 10px 0 0 0;
  color: var(--muted);
  max-width: 52ch;
  line-height: 1.55;
  font-size: 0.98rem;
}

.bct-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: flex-end;
}

.bct-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 12px;
  border-radius: 999px;
  font-size: 0.78rem;
  font-weight: 600;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--muted);
}

.bct-chip.ok { background: var(--sage-soft); color: var(--sage); border-color: oklch(0.74 0.07 148 / 0.35); }
.bct-chip.warn { background: var(--accent-soft); color: var(--amber); border-color: oklch(0.82 0.11 85 / 0.35); }
.bct-chip.live { background: var(--accent-soft); color: var(--accent); border-color: oklch(0.70 0.13 42 / 0.35); }

.bct-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 20px;
}

.bct-panel-title {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin: 0 0 14px 0;
  font-weight: 700;
}

.bct-review-card {
  background: var(--surface-raised);
  border: 1px solid var(--border);
  border-radius: calc(var(--radius) + 2px);
  padding: 22px 24px;
}

.bct-review-title {
  font-family: "Archivo", sans-serif;
  font-size: 1.15rem;
  font-weight: 700;
  margin: 10px 0 14px 0;
  color: var(--text);
}

.bct-review-body {
  font-family: "Source Serif 4", Georgia, serif;
  font-size: 1.02rem;
  line-height: 1.65;
  color: oklch(0.88 0.02 88);
  margin: 0;
}

.bct-stars {
  letter-spacing: 0.08em;
  color: var(--amber);
  font-size: 1.05rem;
}

.bct-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.bct-badge.stub { background: var(--surface); color: var(--muted); border: 1px solid var(--border); }
.bct-badge.faiss { background: var(--accent-soft); color: var(--accent); border: 1px solid oklch(0.70 0.13 42 / 0.35); }
.bct-badge.llm { background: var(--sage-soft); color: var(--sage); border: 1px solid oklch(0.74 0.07 148 / 0.35); }

.bct-info-block {
  color: oklch(0.86 0.02 88);
  line-height: 1.65;
  font-size: 0.95rem;
}

.bct-info-block p { margin: 0 0 12px 0; }
.bct-info-block p:last-child { margin-bottom: 0; }
.bct-info-block strong { color: var(--text); font-weight: 600; }
.bct-info-block a { color: var(--accent); text-decoration: none; }
.bct-info-block a:hover { text-decoration: underline; }

.bct-step-list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.bct-step-list li {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 14px;
  align-items: start;
  padding: 14px 16px;
  background: var(--surface-raised);
  border: 1px solid var(--border);
  border-radius: 12px;
}

.bct-step-num {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 0.78rem;
  font-weight: 700;
  flex-shrink: 0;
}

.bct-step-body strong {
  display: block;
  color: var(--text);
  font-size: 0.92rem;
  margin-bottom: 4px;
}

.bct-step-body span {
  color: var(--muted);
  font-size: 0.88rem;
  line-height: 1.5;
}

.bct-callout {
  padding: 14px 16px;
  border-radius: 12px;
  border: 1px solid oklch(0.74 0.07 148 / 0.35);
  background: var(--sage-soft);
  color: oklch(0.88 0.03 148);
  font-size: 0.9rem;
  line-height: 1.55;
  margin: 0;
}

.bct-history-item {
  padding: 12px 0;
  border-bottom: 1px solid var(--border);
}

.bct-history-item:last-child { border-bottom: none; }

.bct-history-meta {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 0.78rem;
  color: var(--muted);
  margin-bottom: 6px;
}

.bct-history-text {
  font-family: "Source Serif 4", Georgia, serif;
  font-size: 0.92rem;
  line-height: 1.55;
  color: oklch(0.86 0.02 88);
  margin: 0;
}

.bct-metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
}

.bct-metric {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
}

.bct-metric-label {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 6px;
}

.bct-metric-value {
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--text);
}

.bct-category-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.92rem;
}

.bct-category-row:last-child { border-bottom: none; }

.bct-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 8px;
}

.bct-dot.on { background: var(--sage); }
.bct-dot.off { background: oklch(0.45 0.02 52); }

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _esc(text: str) -> str:
    return html.escape(str(text or ""))


def stars(rating: float | int) -> str:
    r = max(0, min(5, int(round(float(rating)))))
    filled = "★" * r
    empty = "☆" * (5 - r)
    return f'<span class="bct-stars" aria-label="{r} out of 5">{filled}{empty}</span>'


def chip(label: str, tone: str = "neutral") -> str:
    cls = f"bct-chip {tone}" if tone != "neutral" else "bct-chip"
    return f'<span class="{cls}">{_esc(label)}</span>'


def header_block(
    title: str,
    subtitle: str,
    chips: list[tuple[str, str]],
    *,
    kicker: str = "Amazon Reviews 2023 · Hackathon demo",
) -> None:
    chips_html = "".join(chip(text, tone) for text, tone in chips)
    st.markdown(
        f"""
<div class="bct-header">
  <div>
    <p class="bct-kicker">{_esc(kicker)}</p>
    <h1 class="bct-title">{_esc(title)}</h1>
    <p class="bct-subtitle">{_esc(subtitle)}</p>
  </div>
  <div class="bct-chip-row">{chips_html}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def panel(title: str, body_html: str) -> None:
    st.markdown(
        f"""
<div class="bct-panel">
  <p class="bct-panel-title">{_esc(title)}</p>
  {body_html}
</div>
        """,
        unsafe_allow_html=True,
    )


def _mode_badge(mode: str) -> tuple[str, str]:
    m = (mode or "stub").lower()
    if m == "llm":
        return "llm", "Claude"
    if "faiss" in m:
        return "faiss", "FAISS + heuristic"
    return "stub", "Baseline"


def review_card(title: str, text: str, rating: float | int, mode: str) -> None:
    badge_cls, badge_label = _mode_badge(mode)
    st.markdown(
        f"""
<div class="bct-review-card">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
    {stars(rating)}
    <span class="bct-badge {badge_cls}">{_esc(badge_label)}</span>
  </div>
  <div class="bct-review-title">{_esc(title)}</div>
  <p class="bct-review-body">{_esc(text)}</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def history_list(items: list[dict]) -> None:
    if not items:
        st.markdown('<p style="color:var(--muted);margin:0;">No prior reviews for this user.</p>', unsafe_allow_html=True)
        return
    rows = []
    for i, item in enumerate(items, 1):
        rows.append(
            f"""
<div class="bct-history-item">
  <div class="bct-history-meta">
    <span>Review {i} · {_esc(item.get('title') or 'Untitled')}</span>
    <span>{stars(item.get('rating', 0))}</span>
  </div>
  <p class="bct-history-text">{_esc(item.get('text', ''))}</p>
</div>
            """
        )
    panel("Past reviews", "".join(rows))


def metric_grid(metrics: list[tuple[str, str]]) -> None:
    cells = "".join(
        f"""
<div class="bct-metric">
  <div class="bct-metric-label">{_esc(label)}</div>
  <div class="bct-metric-value">{_esc(value)}</div>
</div>
        """
        for label, value in metrics
    )
    st.markdown(f'<div class="bct-metric-grid">{cells}</div>', unsafe_allow_html=True)


def info_block(body_html: str) -> None:
    st.markdown(f'<div class="bct-info-block">{body_html}</div>', unsafe_allow_html=True)


def pipeline_steps(steps: list[tuple[str, str]]) -> None:
    items = []
    for i, (title, desc) in enumerate(steps, 1):
        items.append(
            f"<li><span class='bct-step-num'>{i}</span>"
            f"<div class='bct-step-body'><strong>{_esc(title)}</strong>"
            f"<span>{_esc(desc)}</span></div></li>"
        )
    st.markdown(f"<ol class='bct-step-list'>{''.join(items)}</ol>", unsafe_allow_html=True)


def callout(text: str) -> None:
    st.markdown(f'<p class="bct-callout">{_esc(text)}</p>', unsafe_allow_html=True)


def category_status_row(name: str, reviews: int | None, users: int | None) -> str:
    if reviews is None:
        return f"""
<div class="bct-category-row">
  <span><span class="bct-dot off"></span>{_esc(name)}</span>
  <span style="color:var(--muted);">not loaded</span>
</div>
        """
    return f"""
<div class="bct-category-row">
  <span><span class="bct-dot on"></span>{_esc(name)}</span>
  <span style="color:var(--muted);">{reviews:,} reviews · {users:,} users</span>
</div>
    """
