import streamlit as st  
from query import run_query

st.set_page_config(
    page_title="AI Research Assistant",
    layout="wide"
)

st.title("📚 AI Research Assistant")
st.caption("Document-grounded answers using RAG")

question = st.text_input(
    "Enter your research question",
    placeholder="e.g. Explain CNN-based deepfake detection methods"
)

if st.button("Search") and question:
    with st.spinner("Analyzing documents..."):
        result = run_query(question)

    st.subheader("🧠 Direct Answer")
    st.write(result.answerr)

    st.subheader("📌 Key Points")
    for kp in result.key_points:
        st.markdown(f"- {kp}")

    st.subheader("📖 Evidence")
    for paper, text in result.evidence.items():
        st.markdown(f"**{paper}**")
        st.write(text)

    st.subheader("⚠️ Limitations")
    st.write(result.limitations)

    st.subheader("📚 References")
    for ref in result.references:
        st.markdown(f"- {ref}")
