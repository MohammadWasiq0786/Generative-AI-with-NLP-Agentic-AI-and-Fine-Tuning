import streamlit as st
from workflows.langgraph_router import autonomous_pipeline
from utils.graphviz_exporter import export_to_svg
from PIL import Image
import base64

st.set_page_config(page_title="Knowledge Graph Builder", layout="wide")
st.markdown("""
    <style>
        .title {text-align: center; font-size: 36px; font-weight: bold; color: #4A90E2; margin-bottom: 20px;}
        .desc {text-align: center; font-size: 16px; color: #333; margin-bottom: 30px;}
        .footer {text-align: center; font-size: 13px; color: grey; margin-top: 40px;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class='title'>🧠 Multi-Agent Knowledge Graph Builder</div>
    <div class='desc'>Get detailed, AI-generated mind maps and learning roadmaps instantly</div>
""", unsafe_allow_html=True)

topic = st.text_input("🎯 Enter a question or topic (e.g., 'Roadmap for Machine Learning')")

if topic:
    with st.spinner("🔎 Generating your detailed knowledge graph..."):
        graph = autonomous_pipeline(topic)
        svg_path = export_to_svg(graph)
        st.success("✅ Graph generated successfully!")
        st.image(svg_path, caption=f"📌 Visual Knowledge Map for: {topic}", use_column_width=True)

        # Optional: Add download link
        with open(svg_path, "rb") as f:
            svg_data = f.read()
            b64 = base64.b64encode(svg_data).decode()
            href = f'<a href="data:image/svg+xml;base64,{b64}" download="knowledge_graph.svg">📥 Download SVG</a>'
            st.markdown(href, unsafe_allow_html=True)
