import streamlit as st
import pandas as pd
from pathlib import Path
from engine import run_query
from groq_client import MODELS
from utils import build_schema

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

    show_code = st.toggle("Show generated code", value=False)
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
            if msg.get("type") == "chart":
                st.image(msg["content"])
            elif msg.get("type") == "error":
                st.error(msg["content"])
            else:
                st.write(msg["content"])

            if show_code and msg.get("code") and msg["role"] == "assistant":
                with st.expander("Code"):
                    st.code(msg["code"], language="python")

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

            if result["type"] == "chart":
                st.image(result["content"])
            elif result["type"] == "error":
                st.error(result["content"])
            else:
                st.write(result["content"])

            if show_code:
                with st.expander("Generated Code"):
                    st.code(result.get("code", ""), language="python")

        st.session_state.history.append({
            "role": "assistant",
            "type": result["type"],
            "content": result["content"],
            "code": result.get("code", "")
        })

        st.session_state.llm_history.append({"role": "user", "content": question})
        st.session_state.llm_history.append({
            "role": "assistant",
            "content": f"```python\n{result.get('code','')}\n```"
        })

# ── Empty State ─────────────────────────────────────────
else:
    st.info("👆 Upload a dataset to start analysis.")   