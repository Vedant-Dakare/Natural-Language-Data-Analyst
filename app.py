import streamlit as st
import pandas as pd
from engine import run_query
from groq_client import MODELS
from utils import build_schema

st.set_page_config(
    page_title="AI Data Analyst (Groq)",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst")
st.caption("Powered by Groq — free, fast, Say Fuck to OPENAI.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    selected_label = st.selectbox("Model", list(MODELS.keys()))
    model = MODELS[selected_label]
    show_code = st.toggle("Show generated code", value=False)
    show_schema = st.toggle("Show DataFrame schema", value=False)

    st.divider()
    st.markdown("**Try these questions:**")
    examples = [
        "Show the first 10 rows",
        "How many missing values are there?",
        "Plot a bar chart of sales by category",
        "What is the average revenue per month?",
        "Which 5 products have the highest total sales?",
        "Plot a histogram of the price column",
        "Show a correlation heatmap",
        "Summarize this dataset in plain English",
        "Are there any outliers in the revenue column?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state["prefill"] = ex

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.history = []
        st.session_state.llm_history = []
        st.rerun()

# ── File upload ────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload your data file", type=["csv", "xlsx", "xls"])

if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    # Top-level metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing values", int(df.isnull().sum().sum()))
    c4.metric("Numeric cols", int((df.dtypes != "object").sum()))

    if show_schema:
        with st.expander("DataFrame schema"):
            st.code(build_schema(df))

    with st.expander("Data preview (first 20 rows)"):
        st.dataframe(df.head(20), use_container_width=True)

    st.divider()

    # ── Session state ────────────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []      # display messages
    if "llm_history" not in st.session_state:
        st.session_state.llm_history = []  # messages sent to LLM

    # ── Render previous messages ─────────────────────────────────────────────
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            if msg.get("type") == "chart":
                st.image(msg["content"])
            elif msg.get("type") == "error":
                st.error(msg["content"])
            else:
                st.write(msg["content"])
            if show_code and msg.get("code") and msg["role"] == "assistant":
                with st.expander("Generated code"):
                    st.code(msg["code"], language="python")

    # ── New question ─────────────────────────────────────────────────────────
    prefill = st.session_state.pop("prefill", "")
    question = st.chat_input("Ask anything about your data...")
    if prefill:
        question = prefill

    if question:
        # Show user message
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Run query
        with st.chat_message("assistant"):
            with st.spinner(f"Groq is thinking..."):
                result = run_query(
                    df=df,
                    question=question,
                    history=st.session_state.llm_history,
                    model=model,
                )

            # Render result
            if result["type"] == "chart":
                st.image(result["content"])
            elif result["type"] == "error":
                st.error(result["content"])
            else:
                st.write(result["content"])

            if show_code:
                with st.expander("Generated code"):
                    st.code(result.get("code", ""), language="python")

        # Save to histories
        st.session_state.history.append({
            "role": "assistant",
            "type": result["type"],
            "content": result["content"],
            "code": result.get("code", ""),
        })
        # For multi-turn: add the Q+A to llm_history
        st.session_state.llm_history.append({"role": "user", "content": question})
        st.session_state.llm_history.append({
            "role": "assistant",
            "content": f"```python\n{result.get('code','')}\n```"
        })

else:
    st.info("👆 Upload a CSV or Excel file above to get started.")