import streamlit as st
import pandas as pd
from PIL import Image
import io
import subprocess
import os

# ========== 🔧 Import your backend functions ==========
from shafts_agent import *

st.set_page_config(
    page_title="SHAFTS Molecular Similarity Agent",
    page_icon="🧪",
    layout="wide")

# ----------------------- Custom CSS ------------------------
st.markdown("""
<style>
    body {
        background: linear-gradient(to bottom right, #f7f8fa, #eef1f5);
    }
    .main > div {
        padding: 2rem 4rem;
    }
    h1 {
        text-align: center;
        background: -webkit-linear-gradient(90deg, #0072ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .block-container {
        padding-top: 1rem !important;
    }
    .stTextArea textarea {
        border-radius: 10px !important;
    }
    .stButton button {
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        color: white;
        border-radius: 10px;
        font-size: 1.05rem;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        transform: scale(1.02);
        background: linear-gradient(90deg, #00c6ff, #0072ff);
    }
    .stCheckbox label {
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>🧬</h2>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Settings & Tips")
    st.info(
        """
        💬 **Prompt Examples**
        - Compare aspirin to lab compounds  
        - Find top 10 hits similar to paracetamol  
        - Compare similarity between a SMILES and database  
        """
    )
    st.markdown("---")
    st.markdown("👨‍🔬 *Developed by Corey Jiang*")

# ----------------------- Main Title ------------------------
st.title("🧬 SHAFTS 3D Molecular Similarity Agent")
st.markdown(
    """
    <div style="background-color:#f0f8ff; border-radius:10px; padding:1rem 1.5rem; margin-bottom:1rem;">
    <h3>🔍 Welcome!</h3>
    This interactive agent uses <b>SHAFTS</b> to compute 3D molecular similarity between a query compound and your compound database.
    </div>
    """, unsafe_allow_html=True
)
st.markdown(
    """
    💡 <b>You can ask naturally, for example:</b>
    - `Compare similarity between aspirin and our mini lab compounds. Show top 5 hits.`
    - `Now compare similarity between O=C(C)Oc1ccccc1C(=O)O and our database.`
    - `Find top 10 compounds similar to paracetamol.`

    The agent will:
    1. Parse your prompt to extract molecular info  
    2. Convert your query molecule into 3D format (mol2)  
    3. Run **Cynthia (SHAFTS)** for 3D similarity  
    4. Parse and display results  
    5. Optionally visualize the best hit molecule 🌟  
    """, unsafe_allow_html=True
)

# ----------------------- Inputs ----------------------------
user_prompt = st.text_area(
    "💬 Type your query prompt here:",
    height=120,
    placeholder="e.g. Compare similarity between aspirin and our mini lab compounds. Show top 5 hits.")
# visualize_option = st.checkbox("🧠 Visualize top hit molecule", value=True)
# run_button = st.button("🚀 Run Similarity Search")

col1, col2 = st.columns([1, 1])
with col1:
    visualize_option = st.checkbox("🧠 Visualize top hit molecule", value=True)
with col2:
    run_button = st.button("🚀 Run Similarity Search")

# ----------------------- Main Logic ------------------------
if run_button:
    if not user_prompt.strip():
        st.warning("⚠️ Please enter a valid prompt.")
        st.stop()

    with st.spinner("🤖 Processing your prompt using LLM reasoning..."):
        info = extract_query_info(user_prompt)
        query_smiles = info.get("query_smiles", "").strip()
        query_name = info.get("query_name", "").strip()
        num_shown = info.get("num_shown", 10)

        # Step 1: resolve missing info
        if not query_smiles and not query_name:
            st.error("❌ Neither SMILES nor compound name provided in your prompt.")
            st.stop()

        if query_smiles and not query_name:
            query_name = get_compound_name_from_smiles(query_smiles) or "query"

        if query_name and not query_smiles:
            query_smiles = get_smiles_from_name(query_name)
        
        st.write(f"✅ Identified molecule: **{query_name}**")
        st.code(query_smiles, language="text")

    with st.spinner("🧩 Generating 3D structure and running SHAFTS..."):
        query_sdf = smiles_to_sdf(query_smiles, query_name)
        query_mol2 = f"{query_name}.mol2"

        # Convert sdf → mol2
        subprocess.run(["obabel", query_sdf, "-O", query_mol2], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Run Cynthia
        res = run_cynthia(
            query_path=query_mol2,
            target_path="random_mini_db.sdf",
            cynthia_exe="Cynthia",
            out_prefix=f"results\\{query_name}_vs_db",
            timeout=100
        )

        st.success("✅ SHAFTS finished successfully!")
        st.write("**Output Files:**", res["produced_files"])

    with st.spinner("📊 Parsing similarity results..."):
        output_prefix = f"results\\{query_name}_vs_db"
        results = parse_cynthia_output(output_prefix)

        if results.empty:
            st.error("No valid results parsed from Cynthia output.")
            st.stop()

        st.subheader("📈 Top Hits")
        st.dataframe(results.head(num_shown), width='stretch')

    if visualize_option:
        with st.spinner("🧬 Visualizing top hit molecule..."):
            img = visualize_best_hit(results, db_file="random_mini_db.sdf", top_n=num_shown, save_path="best_hit.png")

            if isinstance(img, Image.Image):
                st.image(img, caption="Top Hit Molecule (Best HybridScore)", width='content')
            elif isinstance(img, str) and os.path.exists(img):
                st.image(Image.open(img), caption="Top Hit Molecule (Best HybridScore)")
            else:
                st.warning("⚠️ Unable to visualize best hit molecule.")


st.markdown("---")
st.markdown("💻 *Developed with ❤️ using Streamlit + SHAFTS + RDKit + DeepSeek reasoning*")