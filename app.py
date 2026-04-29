import base64
import html
import io
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from engine import run_query
from groq_client import MODELS
from utils import build_schema


def _svg_avatar(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


USER_AVATAR = _svg_avatar(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none">
            <defs>
                <linearGradient id="userGrad" x1="10" y1="10" x2="54" y2="54" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#55d6c2"/>
                    <stop offset="1" stop-color="#1f7f8c"/>
                </linearGradient>
            </defs>
            <circle cx="32" cy="32" r="30" fill="url(#userGrad)"/>
            <circle cx="32" cy="25" r="10" fill="rgba(255,255,255,0.95)"/>
            <path d="M16 51c3.2-9 11-14 16-14s12.8 5 16 14" fill="rgba(255,255,255,0.95)"/>
            <circle cx="24" cy="24" r="2" fill="#174b55"/>
            <circle cx="40" cy="24" r="2" fill="#174b55"/>
            <path d="M25 29c2 1.8 3.9 2.7 7 2.7s5-.9 7-2.7" stroke="#174b55" stroke-width="2.4" stroke-linecap="round"/>
        </svg>
        """
)

ASSISTANT_AVATAR = _svg_avatar(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" fill="none">
            <defs>
                <linearGradient id="aiGrad" x1="12" y1="12" x2="52" y2="52" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#ffbf5e"/>
                    <stop offset="1" stop-color="#f46f4f"/>
                </linearGradient>
            </defs>
            <circle cx="32" cy="32" r="30" fill="url(#aiGrad)"/>
            <rect x="20" y="21" width="24" height="20" rx="10" fill="rgba(255,255,255,0.95)"/>
            <circle cx="27" cy="31" r="2.5" fill="#8e4025"/>
            <circle cx="37" cy="31" r="2.5" fill="#8e4025"/>
            <path d="M26 39c2.2 1.8 4.5 2.7 6 2.7s3.8-.9 6-2.7" stroke="#8e4025" stroke-width="2.2" stroke-linecap="round"/>
            <path d="M32 10l1.7 4.8 4.8 1.7-4.8 1.7L32 23l-1.7-4.8-4.8-1.7 4.8-1.7z" fill="#fff3cf"/>
        </svg>
        """
)

STATE_DIR = Path(__file__).with_name(".app_state")
STATE_FILE = STATE_DIR / "state.json"
UPLOAD_FILE = STATE_DIR / "upload.bin"


def _load_persisted_state() -> dict:
    if not STATE_FILE.exists():
        return {"history": [], "llm_history": [], "upload_name": None}
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"history": [], "llm_history": [], "upload_name": None}

    history = payload.get("history", [])
    llm_history = payload.get("llm_history", [])
    upload_name = payload.get("upload_name")
    if not isinstance(history, list):
        history = []
    if not isinstance(llm_history, list):
        llm_history = []
    if not isinstance(upload_name, str):
        upload_name = None
    return {
        "history": history,
        "llm_history": llm_history,
        "upload_name": upload_name,
    }


def _save_persisted_state(history: list[dict], llm_history: list[dict], upload_name: str | None) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "history": history,
        "llm_history": llm_history,
        "upload_name": upload_name,
    }
    STATE_FILE.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def _clear_persisted_state() -> None:
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    if UPLOAD_FILE.exists():
        UPLOAD_FILE.unlink()


def _read_uploaded_dataframe(file_name: str, blob: bytes) -> pd.DataFrame:
    stream = io.BytesIO(blob)
    if file_name.lower().endswith(".csv"):
        return pd.read_csv(stream)
    return pd.read_excel(stream)


def _persist_uploaded_file(blob: bytes) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_FILE.write_bytes(blob)


def _restore_uploaded_dataframe(upload_name: str | None) -> pd.DataFrame | None:
    if not upload_name or not UPLOAD_FILE.exists():
        return None
    try:
        return _read_uploaded_dataframe(upload_name, UPLOAD_FILE.read_bytes())
    except Exception:
        return None


def _to_dataframe(payload) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        return payload
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        # list-like dict values become columns naturally
        try:
            return pd.DataFrame(payload)
        except Exception:
            return pd.DataFrame([payload])
    return pd.DataFrame()


def _render_result_header(title: str, description: str, result_type: str) -> None:
    icon_by_type = {
        "table": "data_usage",
        "chart": "insights",
        "graph": "hub",
        "error": "warning",
    }
    icon = icon_by_type.get(result_type, "auto_graph")
    safe_title = html.escape(title)
    safe_desc = html.escape(description)
    safe_type = html.escape(result_type.title())

    st.markdown(
        f"""
        <div class="result-shell">
            <div class="result-topline">
                <span class="result-icon material-symbols-rounded">{icon}</span>
                <span class="result-badge">{safe_type}</span>
            </div>
            <div class="result-title">{safe_title}</div>
            <p class="result-description">{safe_desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _first_numeric_col(df: pd.DataFrame) -> str | None:
    numeric_cols = list(df.select_dtypes(include="number").columns)
    return numeric_cols[0] if numeric_cols else None


def _make_plotly_chart(chart_df: pd.DataFrame, chart_type: str, x_label: str, y_label: str):
    x_col = chart_df.columns[0] if len(chart_df.columns) > 0 else None
    y_col = chart_df.columns[1] if len(chart_df.columns) > 1 else _first_numeric_col(chart_df)
    palette = ["#4bc0a1", "#5ec7d7", "#f2a65a", "#96d06f", "#f77f7f"]

    if chart_type == "histogram":
        value_col = _first_numeric_col(chart_df)
        if not value_col:
            return None
        fig = px.histogram(
            chart_df,
            x=value_col,
            nbins=min(24, max(8, int(len(chart_df) ** 0.5))),
            color_discrete_sequence=[palette[0]],
        )
    elif chart_type == "line":
        if x_col and y_col and x_col != y_col:
            fig = px.line(chart_df, x=x_col, y=y_col, markers=True, color_discrete_sequence=[palette[1]])
        elif y_col:
            fig = px.line(chart_df, y=y_col, markers=True, color_discrete_sequence=[palette[1]])
        else:
            return None
    else:
        if x_col and y_col and x_col != y_col:
            fig = px.bar(chart_df, x=x_col, y=y_col, color_discrete_sequence=[palette[2]])
        elif y_col:
            fig = px.bar(chart_df, y=y_col, color_discrete_sequence=[palette[2]])
        else:
            return None

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(16, 31, 38, 0.45)",
        margin=dict(l=12, r=12, t=24, b=8),
        font=dict(color="#d7e6ec"),
        xaxis_title=x_label or None,
        yaxis_title=y_label or None,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(105, 138, 151, 0.25)", zeroline=False)
    return fig


def _render_result(result: dict, show_raw: bool = False) -> None:
    if result.get("type") == "error":
        st.error(result.get("content", "Unknown error"))
        if show_raw and result.get("raw"):
            with st.expander("Raw Model Output"):
                st.code(result["raw"], language="json")
        return

    content = result.get("content", {})
    if not isinstance(content, dict):
        st.write(content)
        return

    title = content.get("title", "Result")
    description = content.get("description", "")
    style = content.get("style", {}) if isinstance(content.get("style", {}), dict) else {}

    result_type = result.get("type", "table")
    _render_result_header(title=title, description=description or "Insight generated from your dataset.", result_type=result_type)

    data = content.get("data", {})

    if result_type == "table":
        table_df = _to_dataframe(data)
        if table_df.empty:
            st.info("No table rows were returned.")
        else:
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Rows", f"{table_df.shape[0]:,}")
            tc2.metric("Columns", table_df.shape[1])
            tc3.metric("Missing", int(table_df.isna().sum().sum()))
            st.dataframe(table_df, use_container_width=True, hide_index=True)

    elif result_type == "chart":
        chart_df = _to_dataframe(data)
        chart_type = style.get("chart_type", "bar")
        x_label = style.get("x_label", "")
        y_label = style.get("y_label", "")

        if chart_df.empty or chart_df.shape[1] == 0:
            st.info("No chart data was returned.")
        else:
            fig = _make_plotly_chart(
                chart_df=chart_df,
                chart_type=chart_type,
                x_label=x_label,
                y_label=y_label,
            )
            if fig is None:
                st.info("Chart could not be generated from this data shape.")
            else:
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        if x_label or y_label:
            st.caption(f"X: {x_label or '-'} | Y: {y_label or '-'}")
        if style.get("highlight"):
            st.markdown(f"<div class='insight-pill'>Key Insight: {html.escape(str(style['highlight']))}</div>", unsafe_allow_html=True)

    elif result_type == "graph":
        graph_data = content.get("graph", {}) if isinstance(content.get("graph", {}), dict) else {}
        nodes = graph_data.get("nodes", []) if isinstance(graph_data.get("nodes", []), list) else []
        edges = graph_data.get("edges", []) if isinstance(graph_data.get("edges", []), list) else []

        if nodes:
            dot_lines = [
                "digraph G {",
                "  graph [bgcolor=transparent, pad=0.35, nodesep=0.55, ranksep=0.85, splines=true, overlap=false];",
                "  node [shape=ellipse, style=filled, color=\"#2d5f70\", fillcolor=\"#eaf8f5\", fontname=\"Manrope\", fontsize=12];",
                "  edge [color=\"#79aabd\", fontcolor=\"#d7e6ec\", arrowsize=0.8, penwidth=1.2, fontsize=10, fontname=\"Manrope\"];",
            ]
            for n in nodes:
                node_id = str(n.get("id", n.get("label", "node")))
                node_label = str(n.get("label", node_id))
                safe_id = node_id.replace('"', "'")
                safe_label = node_label.replace('"', "'")
                is_root = "dataset" in node_label.lower()
                if is_root:
                    dot_lines.append(
                        f'  "{safe_id}" [label="{safe_label}", fillcolor="#ffe2bc", color="#da8f4f", penwidth=1.4];'
                    )
                else:
                    dot_lines.append(f'  "{safe_id}" [label="{safe_label}"];')

            for e in edges:
                src = str(e.get("source", ""))
                dst = str(e.get("target", ""))
                if src and dst:
                    safe_src = src.replace('"', "'")
                    safe_dst = dst.replace('"', "'")
                    safe_edge = str(e.get("label", "")).replace('"', "'")
                    dot_lines.append(f'  "{safe_src}" -> "{safe_dst}" [label="{safe_edge}"];')

            dot_lines.append("}")
            dot = "\n".join(dot_lines)

            st.graphviz_chart(dot, use_container_width=True)
        else:
            st.info("No graph nodes were returned.")

        if edges:
            with st.expander("Relationships"):
                st.dataframe(_to_dataframe(edges), use_container_width=True, hide_index=True)

    if show_raw and result.get("raw"):
        with st.expander("Raw Model Output"):
            st.code(result["raw"], language="json")

# ── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)
# ── Custom CSS (loaded from external file) ─────────────
css_file = Path(__file__).with_name("styles.css")
st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

persisted_state = _load_persisted_state()
if "history" not in st.session_state:
    st.session_state.history = persisted_state["history"]
if "llm_history" not in st.session_state:
    st.session_state.llm_history = persisted_state["llm_history"]
if "upload_name" not in st.session_state:
    st.session_state.upload_name = persisted_state["upload_name"]

# ── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## AnalystAI")

    st.markdown("### ⚙️ Settings")
    selected_label = st.selectbox("Model", list(MODELS.keys()))
    model = MODELS[selected_label]

    show_code = st.toggle("Show raw model JSON", value=False)
    show_schema = st.toggle("Show schema", value=False)

    st.markdown("---")
    st.markdown("### 💡 Examples")

    examples = [
        "Create a relationship graph of all key columns in this dataset",
        "Show the hierarchy between dataset, entities, and attributes",
        "Build a graph of strongest correlations between numeric features",
        "Which columns are directly related to revenue? Show as a graph",
        "Generate an entity-relationship style map from this data",
        "Show dependency links between category, product, and sales fields",
        "Identify hub columns with the most relationships",
        "Build a network graph of variables and their connections",
        "Show clusters of related columns in a relationship graph"
    ]

    for ex in examples:
        if st.button(ex):
            st.session_state["prefill"] = ex

    st.markdown("---")
    if st.button("🗑 Clear Chat"):
        st.session_state.history = []
        st.session_state.llm_history = []
        st.session_state.upload_name = None
        _clear_persisted_state()
        _save_persisted_state(st.session_state.history, st.session_state.llm_history, st.session_state.upload_name)
        st.rerun()

# ── Header Section ──────────────────────────────────────
st.markdown("""
<div class="title">AI Data Analyst</div>
<p style="text-align:center;color:#9db0ba;">
Upload your dataset to analyze trends, generate charts, and get insights.
</p>
""", unsafe_allow_html=True)

# ── Upload Section ──────────────────────────────────────
uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
df = None

if uploaded is not None:
    upload_name = uploaded.name
    upload_blob = uploaded.getvalue()
    try:
        df = _read_uploaded_dataframe(upload_name, upload_blob)
    except Exception as exc:
        st.error(f"Unable to read uploaded file: {exc}")
    else:
        st.session_state.upload_name = upload_name
        _persist_uploaded_file(upload_blob)
        _save_persisted_state(st.session_state.history, st.session_state.llm_history, st.session_state.upload_name)
else:
    df = _restore_uploaded_dataframe(st.session_state.upload_name)
    if df is not None and st.session_state.upload_name:
        st.caption(f"Restored previous dataset: {st.session_state.upload_name}")

# ── If File Uploaded ────────────────────────────────────
if df is not None:

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", int(df.isnull().sum().sum()))
    c4.metric("Numeric", int((df.dtypes != "object").sum()))

    # Schema
    if show_schema:
        with st.expander("Schema"):
            st.code(build_schema(df))

    # Preview
    with st.expander("Preview"):
        st.dataframe(df.head(20))

    # Show chat history
    for msg in st.session_state.history:
        with st.chat_message(
            msg["role"],
            avatar=ASSISTANT_AVATAR if msg["role"] == "assistant" else USER_AVATAR,
        ):
            if msg["role"] == "assistant":
                _render_result(msg, show_raw=show_code)
            else:
                st.write(msg["content"])

    # Input
    prefill = st.session_state.pop("prefill", "")
    question = st.chat_input("Ask about your data...")

    if prefill:
        question = prefill

    if question:
        st.session_state.history.append({"role": "user", "content": question})

        with st.chat_message("user", avatar=USER_AVATAR):
            st.write(question)

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Thinking..."):
                result = run_query(
                    df=df,
                    question=question,
                    history=st.session_state.llm_history,
                    model=model
                )
            _render_result(result, show_raw=show_code)

        st.session_state.history.append({
            "role": "assistant",
            "type": result["type"],
            "content": result["content"],
            "raw": result.get("raw", "")
        })

        st.session_state.llm_history.append({"role": "user", "content": question})
        st.session_state.llm_history.append({
            "role": "assistant",
            "content": str(result.get("content", {}))
        })
        _save_persisted_state(st.session_state.history, st.session_state.llm_history, st.session_state.upload_name)

# ── Empty State ─────────────────────────────────────────
else:
    st.info("👆 Upload a dataset to start analysis.")   