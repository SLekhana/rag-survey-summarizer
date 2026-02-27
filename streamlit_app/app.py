"""
Streamlit Dashboard
- Document ingestion UI
- Search with mode selector
- Executive summary display
- Evaluation results comparison
- Usage dashboard
"""

import streamlit as st
import requests
import json
import pandas as pd
import time

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Survey Summarizer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RAG Survey Response Summarizer")
st.caption("LLM-powered semantic search & executive summarization | BM25 + FAISS hybrid retrieval")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📥 Ingest", "🔍 Search", "📝 Summarize", "🧪 Evaluate", "📈 Dashboard"])


# ──────────────────────────────────────────────
# TAB 1: INGEST
# ──────────────────────────────────────────────
with tab1:
    st.header("Document Ingestion")
    st.info("Upload survey responses for indexing. Documents are chunked, embedded, and stored in FAISS + ChromaDB.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload JSON")
        uploaded_file = st.file_uploader("Upload survey responses (JSON)", type=["json"])
        if uploaded_file:
            data = json.load(uploaded_file)
            st.write(f"Loaded {len(data.get('documents', []))} documents")
            if st.button("Ingest Documents", type="primary"):
                with st.spinner("Ingesting..."):
                    response = requests.post(f"{API_URL}/ingest", json=data)
                    if response.ok:
                        result = response.json()
                        st.success(f"✅ Ingested {result['ingested']} docs | Failed: {result['failed']} | {result['duration_ms']:.0f}ms")
                    else:
                        st.error(f"Error: {response.text}")

    with col2:
        st.subheader("Sample Data")
        st.code(json.dumps({
            "documents": [
                {"id": "1", "text": "The product quality is excellent but delivery was slow.", "metadata": {"source": "q1"}},
                {"id": "2", "text": "Customer support resolved my issue quickly.", "metadata": {"source": "q1"}},
                {"id": "3", "text": "I love the new features but the UI feels outdated.", "metadata": {"source": "q2"}}
            ]
        }, indent=2), language="json")

        if st.button("Load Sample Data"):
            sample = {
                "documents": [
                    {"id": f"sample_{i}", "text": t, "metadata": {"source": "sample"}}
                    for i, t in enumerate([
                        "The product quality exceeded my expectations. Very satisfied.",
                        "Delivery took too long, over 2 weeks. Very disappointing.",
                        "Customer service was helpful and resolved my issue promptly.",
                        "The pricing is too high compared to competitors.",
                        "Love the user interface, very intuitive and easy to use.",
                        "Technical issues with the app on mobile devices need fixing.",
                        "The onboarding process was smooth and well-guided.",
                        "Would appreciate more customization options.",
                        "The product crashes frequently. Reliability is a major concern.",
                        "Excellent value for money. Would highly recommend."
                    ])
                ]
            }
            response = requests.post(f"{API_URL}/ingest", json=sample)
            if response.ok:
                result = response.json()
                st.success(f"✅ Sample ingested: {result['ingested']} documents")


# ──────────────────────────────────────────────
# TAB 2: SEARCH
# ──────────────────────────────────────────────
with tab2:
    st.header("Semantic Search")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input("Search query", placeholder="What do customers think about delivery?")
    with col2:
        mode = st.selectbox("Retrieval Mode", ["hybrid", "dense", "sparse"])
    with col3:
        top_k = st.slider("Top K", 3, 20, 10)

    if st.button("Search", type="primary") and query:
        start = time.time()
        response = requests.post(f"{API_URL}/search", json={
            "query": query, "top_k": top_k, "mode": mode
        })
        elapsed = (time.time() - start) * 1000

        if response.ok:
            data = response.json()
            st.success(f"Found {len(data['results'])} results in {data['latency_ms']:.0f}ms (mode: {mode})")

            for i, result in enumerate(data["results"], 1):
                with st.expander(f"[{i}] Score: {result['score']:.4f}"):
                    st.write(result["text"])
                    st.caption(f"ID: {result['id'][:12]}...")
        else:
            st.error(response.text)


# ──────────────────────────────────────────────
# TAB 3: SUMMARIZE
# ──────────────────────────────────────────────
with tab3:
    st.header("Executive Summarization")

    col1, col2 = st.columns([2, 1])
    with col1:
        sum_query = st.text_input("Summarization query", placeholder="What are the main themes in customer feedback?")
    with col2:
        use_gpt4 = st.checkbox("Use GPT-4 (stronger)")
        max_themes = st.slider("Max themes", 2, 8, 5)

    sum_mode = st.radio("Retrieval mode", ["hybrid", "dense", "sparse"], horizontal=True)

    if st.button("Summarize", type="primary") and sum_query:
        with st.spinner("Retrieving and generating summary..."):
            response = requests.post(f"{API_URL}/summarize", json={
                "query": sum_query,
                "mode": sum_mode,
                "use_strong_model": use_gpt4,
                "max_themes": max_themes
            })

        if response.ok:
            data = response.json()

            # Executive summary
            st.subheader("📋 Executive Summary")
            st.info(data["executive_summary"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Responses Analyzed", data["total_responses_analyzed"])
            col2.metric("Latency", f"{data['latency_ms']:.0f}ms")
            col3.metric("Model", data["model_used"])

            if data.get("rouge_scores"):
                rouge = data["rouge_scores"]
                st.caption(f"ROUGE: R1={rouge.get('rouge1',0):.3f} | R2={rouge.get('rouge2',0):.3f} | RL={rouge.get('rougeL',0):.3f}")

            # Themes
            st.subheader("🏷️ Themes Detected")
            for theme in data.get("themes", []):
                with st.expander(f"**{theme['theme']}** (confidence: {theme['confidence']:.0%})"):
                    st.write(theme["summary"])
                    if theme.get("supporting_responses"):
                        st.caption("Supporting responses:")
                        for resp in theme["supporting_responses"]:
                            st.write(f"> {resp}")
        else:
            st.error(response.text)


# ──────────────────────────────────────────────
# TAB 4: EVALUATE
# ──────────────────────────────────────────────
with tab4:
    st.header("Controlled Retrieval Experiments")
    st.caption("Compare sparse vs dense vs hybrid vs TF-IDF baseline. Target: ≥20% improvement over TF-IDF.")

    eval_query = st.text_input("Evaluation query", placeholder="What are customer pain points?")
    gt_themes = st.text_area("Ground truth themes (one per line)",
                              placeholder="delivery speed\nproduct quality\ncustomer support")

    if st.button("Run Experiment", type="primary") and eval_query and gt_themes:
        themes_list = [t.strip() for t in gt_themes.strip().split("\n") if t.strip()]

        with st.spinner("Running experiments across all modes..."):
            response = requests.post(f"{API_URL}/evaluate", json={
                "query": eval_query,
                "ground_truth_themes": themes_list,
                "top_k": 10
            })

        if response.ok:
            data = response.json()
            st.success(f"Best mode: **{data['best_mode']}**")

            results_df = pd.DataFrame([
                {
                    "Mode": r["mode"],
                    "ROUGE-1": f"{r['rouge_1']:.3f}",
                    "ROUGE-2": f"{r['rouge_2']:.3f}",
                    "ROUGE-L": f"{r['rouge_l']:.3f}",
                    "Theme Accuracy": f"{r['theme_detection_accuracy']:.1%}",
                    "Hallucination": f"{r['hallucination_score']:.3f}",
                    "Latency (ms)": f"{r['latency_ms']:.0f}"
                }
                for r in data["results"]
            ])
            st.dataframe(results_df, use_container_width=True)

            # Visual comparison
            theme_acc_data = {r["mode"]: r["theme_detection_accuracy"] for r in data["results"]}
            if theme_acc_data:
                chart_df = pd.DataFrame(list(theme_acc_data.items()), columns=["Mode", "Theme Accuracy"])
                st.bar_chart(chart_df.set_index("Mode"))
        else:
            st.error(response.text)


# ──────────────────────────────────────────────
# TAB 5: DASHBOARD
# ──────────────────────────────────────────────
with tab5:
    st.header("Usage Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Refresh Stats"):
            response = requests.get(f"{API_URL}/dashboard")
            if response.ok:
                stats = response.json()
                st.metric("Total Requests", stats["total_requests"])
                st.metric("Total Documents", stats["total_documents"])
                st.metric("Avg Latency", f"{stats['avg_latency_ms']:.0f}ms")

                if stats["requests_by_mode"]:
                    st.subheader("Requests by Retrieval Mode")
                    mode_df = pd.DataFrame(list(stats["requests_by_mode"].items()), columns=["Mode", "Count"])
                    st.bar_chart(mode_df.set_index("Mode"))

    with col2:
        health_response = requests.get(f"{API_URL}/health")
        if health_response.ok:
            h = health_response.json()
            st.subheader("System Health")
            st.metric("FAISS Documents", h["faiss_docs"])
            st.metric("ChromaDB Documents", h["chroma_docs"])
            st.metric("Schema Version", h["schema_version"])
            st.success(f"Status: {h['status'].upper()}")
