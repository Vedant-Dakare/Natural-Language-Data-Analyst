import streamlit as st
import pandas as pd
from pathlib import Path
from engine import run_query
from groq_client import MODELS
from utils import build_schema


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

    st.markdown(f"### {title}")
    if description:
        st.caption(description)

    result_type = result.get("type", "table")
    data = content.get("data", {})

    if result_type == "table":
        table_df = _to_dataframe(data)
        if table_df.empty:
            st.info("No table rows were returned.")
        else:
            st.dataframe(table_df, use_container_width=True)

    elif result_type == "chart":
        chart_df = _to_dataframe(data)
        chart_type = style.get("chart_type", "bar")
        x_label = style.get("x_label", "")
        y_label = style.get("y_label", "")

        if chart_df.empty or chart_df.shape[1] == 0:
            st.info("No chart data was returned.")
        elif chart_type == "histogram":
            numeric_cols = chart_df.select_dtypes(include="number").columns
            if len(numeric_cols) == 0:
                st.info("Histogram needs numeric values.")
            else:
                st.bar_chart(chart_df[numeric_cols[0]])
        else:
            if chart_df.shape[1] >= 2:
                chart_df = chart_df.set_index(chart_df.columns[0])
            if chart_type == "line":
                st.line_chart(chart_df)
            else:
                st.bar_chart(chart_df)

        if x_label or y_label:
            st.caption(f"X: {x_label or '-'} | Y: {y_label or '-'}")
        if style.get("highlight"):
            st.info(f"Highlight: {style['highlight']}")

    elif result_type == "graph":
        graph_data = content.get("graph", {}) if isinstance(content.get("graph", {}), dict) else {}
        nodes = graph_data.get("nodes", []) if isinstance(graph_data.get("nodes", []), list) else []
        edges = graph_data.get("edges", []) if isinstance(graph_data.get("edges", []), list) else []

        if nodes:
            dot_lines = ["digraph G {"]
            for n in nodes:
                node_id = str(n.get("id", n.get("label", "node")))
                node_label = str(n.get("label", node_id))
                safe_id = node_id.replace('"', "'")
                safe_label = node_label.replace('"', "'")
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
                st.dataframe(_to_dataframe(edges), use_container_width=True)

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
        "Show the first 10 rows",
        "How many missing values are there?",
        "Plot a bar chart of sales by category",
        "Average revenue per month?",
        "Top 5 products by sales?",
        "Histogram of price column",
        "Correlation heatmap",
        "Summarize dataset",
        "Find outliers in revenue"
    ]

    for ex in examples:
        if st.button(ex):
            st.session_state["prefill"] = ex

    st.markdown("---")
    if st.button("🗑 Clear Chat"):
        st.session_state.history = []
        st.session_state.llm_history = []
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

# ── If File Uploaded ────────────────────────────────────
if uploaded:

    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

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

    # Session state
    if "history" not in st.session_state:
        st.session_state.history = []
    if "llm_history" not in st.session_state:
        st.session_state.llm_history = []

    # Show chat history
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
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

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
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

# ── Empty State ─────────────────────────────────────────
else:
    st.info("👆 Upload a dataset to start analysis.")   