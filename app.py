import streamlit as st
import pandas as pd
from PIL import Image
import io
import subprocess
import os

# ========== 🔧 Import your backend functions ==========
from shafts_agent import *

st.set_page_config(page_title="SHAFTS Molecular Similarity Agent", page_icon="🧪", layout="wide")

st.title("🧬 SHAFTS 3D Molecular Similarity Agent")
st.markdown(
    """
    ### 🔍 Welcome!
    This interactive agent uses **SHAFTS** to compute 3D molecular similarity between a query compound and your compound database.

    💡 You can ask naturally, for example:
    - `Compare similarity between aspirin and our mini lab compounds. Show top 5 hits.`
    - `Now compare similarity between O=C(C)Oc1ccccc1C(=O)O and our database.`
    - `Find top 10 compounds similar to paracetamol.`

    The agent will:
    1. Parse your prompt to extract molecular info  
    2. Convert your query molecule into 3D format (mol2)  
    3. Run **Cynthia (SHAFTS)** for 3D similarity  
    4. Parse and display results  
    5. Optionally visualize the best hit molecule 🌟  
    """
)

user_prompt = st.text_area("💬 Type your query prompt here:", height=120, placeholder="e.g. Compare similarity between aspirin and our mini lab compounds. Show top 5 hits.")
visualize_option = st.checkbox("🧠 Visualize top hit molecule", value=True)
run_button = st.button("🚀 Run Similarity Search")

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
            query_smiles s= get_smiles_from_name(query_name)
        
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
st.markdown("💻 *Developed with ❤️ using Streamlit + SHAFTS + RDKit + LLM reasoning (DeepSeek)*")


# import streamlit as st
# import pandas as pd
# from PIL import Image
# import io
# import subprocess
# import os

# # ========== 🔧 Import your backend functions ==========
# from shafts_agent import *

# st.set_page_config(page_title="SHAFTS Molecular Similarity Agent", page_icon="🧪", layout="wide")

# # Custom CSS for better styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem !important;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .info-box {
#         background-color: #f0f2f6;
#         padding: 1.5rem;
#         border-radius: 10px;
#         border-left: 5px solid #1f77b4;
#         margin-bottom: 2rem;
#     }
#     .result-box {
#         background-color: #e8f4fd;
#         padding: 1.5rem;
#         border-radius: 10px;
#         border: 1px solid #1f77b4;
#         margin: 1rem 0;
#     }
#     .section-header {
#         color: #1f77b4;
#         border-bottom: 2px solid #1f77b4;
#         padding-bottom: 0.5rem;
#         margin-top: 2rem;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Header Section
# st.markdown('<h1 class="main-header">🧬 SHAFTS 3D Molecular Similarity Agent</h1>', unsafe_allow_html=True)

# # Introduction Section
# with st.container():
#     st.markdown("### 🔍 Welcome!")
#     st.markdown("""
#     <div class="info-box">
#     This interactive agent uses <b>SHAFTS</b> to compute 3D molecular similarity between a query compound and your compound database.
#     </div>
#     """, unsafe_allow_html=True)

# # Create two columns for better layout
# col1, col2 = st.columns([2, 1])

# with col1:
#     st.markdown("#### 💡 Example Queries")
#     st.markdown("""
#     - `Compare similarity between aspirin and our mini lab compounds. Show top 5 hits.`
#     - `Now compare similarity between O=C(C)Oc1ccccc1C(=O)O and our database.`
#     - `Find top 10 compounds similar to paracetamol.`
#     """)

# with col2:
#     st.markdown("#### 🎯 What the Agent Does")
#     st.markdown("""
#     1. Parses your prompt for molecular info  
#     2. Converts query to 3D format (mol2)  
#     3. Runs **SHAFTS** for 3D similarity  
#     4. Displays ranked results  
#     5. Visualizes best hit molecule 🌟  
#     """)

# st.markdown("---")

# # Main Input Section
# st.markdown('<div class="section-header">🔬 Query Input</div>', unsafe_allow_html=True)

# # Create columns for input and options
# input_col, options_col = st.columns([3, 1])

# with input_col:
#     user_prompt = st.text_area(
#         "**💬 Enter your similarity query:**", 
#         height=120, 
#         placeholder="e.g., Compare similarity between aspirin and our mini lab compounds. Show top 5 hits.",
#         help="You can use compound names, SMILES notation, or natural language queries"
#     )

# with options_col:
#     st.markdown("#### ⚙️ Options")
#     visualize_option = st.checkbox("Visualize top hit", value=True, help="Show 2D structure of best matching molecule")
#     num_results = st.slider("Results to show", min_value=5, max_value=20, value=10, help="Number of top hits to display")

# run_button = st.button("🚀 Run Similarity Search", type="primary", use_container_width=True)

# # Processing and Results Section
# if run_button:
#     if not user_prompt.strip():
#         st.error("⚠️ Please enter a valid prompt.")
#         st.stop()

#     # Progress tracking
#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     # Step 1: LLM Processing
#     status_text.text("🤖 Processing your prompt using LLM reasoning...")
#     with st.expander("🔍 Prompt Analysis Details", expanded=False):
#         info = extract_query_info(user_prompt)
#         query_smiles = info.get("query_smiles", "").strip()
#         query_name = info.get("query_name", "").strip()
#         num_shown = info.get("num_shown", num_results)

#         # Resolve missing info
#         if not query_smiles and not query_name:
#             st.error("❌ Neither SMILES nor compound name provided in your prompt.")
#             st.stop()

#         if query_smiles and not query_name:
#             query_name = get_compound_name_from_smiles(query_smiles) or "query"

#         if query_name and not query_smiles:
#             query_smiles = get_smiles_from_name(query_name)
        
#         st.success(f"✅ Identified molecule: **{query_name}**")
#         st.code(query_smiles, language="text")
    
#     progress_bar.progress(25)

#     # Step 2: 3D Structure and SHAFTS
#     status_text.text("🧩 Generating 3D structure and running SHAFTS...")
#     with st.spinner("Converting to 3D and calculating similarities..."):
#         try:
#             query_sdf = smiles_to_sdf(query_smiles, query_name)
#             query_mol2 = f"{query_name}.mol2"

#             # Convert sdf → mol2
#             subprocess.run(["obabel", query_sdf, "-O", query_mol2], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#             # Run Cynthia
#             res = run_cynthia(
#                 query_path=query_mol2,
#                 target_path="random_mini_db.sdf",
#                 cynthia_exe="Cynthia",
#                 out_prefix=f"results\\{query_name}_vs_db",
#                 timeout=100
#             )
#             progress_bar.progress(60)
            
#             st.success("✅ SHAFTS finished successfully!")
#             if "produced_files" in res:
#                 with st.expander("📁 Generated Files", expanded=False):
#                     st.write(res["produced_files"])
                    
#         except Exception as e:
#             st.error(f"❌ Error during SHAFTS execution: {str(e)}")
#             st.stop()

#     # Step 3: Parse Results
#     status_text.text("📊 Parsing similarity results...")
#     output_prefix = f"results\\{query_name}_vs_db"
#     results = parse_cynthia_output(output_prefix)

#     if results.empty:
#         st.error("No valid results parsed from Cynthia output.")
#         st.stop()

#     progress_bar.progress(85)

#     # Display Results
#     status_text.text("🎯 Preparing results display...")
#     st.markdown('<div class="section-header">📊 Similarity Results</div>', unsafe_allow_html=True)
    
#     # Results summary
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Total Results", len(results))
#     with col2:
#         best_score = results['HybridScore'].max() if 'HybridScore' in results.columns else 'N/A'
#         st.metric("Best Score", f"{best_score:.3f}" if isinstance(best_score, (int, float)) else best_score)
#     with col3:
#         st.metric("Query Compound", query_name)
    
#     # Results table
#     st.markdown(f"#### Top {num_shown} Hits")
#     st.dataframe(
#         results.head(num_shown), 
#         use_container_width=True,
#         height=400
#     )
    
#     # Download option
#     csv = results.to_csv(index=False)
#     st.download_button(
#         label="📥 Download Full Results as CSV",
#         data=csv,
#         file_name=f"{query_name}_similarity_results.csv",
#         mime="text/csv",
#     )

#     progress_bar.progress(95)

#     # Visualization Section
#     if visualize_option:
#         st.markdown("---")
#         st.markdown('<div class="section-header">🧬 Molecular Visualization</div>', unsafe_allow_html=True)
        
#         with st.spinner("Generating molecular visualization..."):
#             img = visualize_best_hit(results, db_file="random_mini_db.sdf", top_n=num_shown, save_path="best_hit.png")

#             viz_col1, viz_col2 = st.columns([2, 1])
            
#             with viz_col1:
#                 if isinstance(img, Image.Image):
#                     st.image(img, caption=f"Top Hit Molecule - {query_name} Similarity Search", use_column_width=True)
#                 elif isinstance(img, str) and os.path.exists(img):
#                     st.image(Image.open(img), caption=f"Top Hit Molecule - {query_name} Similarity Search", use_column_width=True)
#                 else:
#                     st.warning("⚠️ Unable to visualize best hit molecule.")

#             with viz_col2:
#                 if not results.empty:
#                     best_hit = results.iloc[0]
#                     st.markdown("#### 🏆 Best Match Info")
#                     st.write(f"**Name:** {best_hit.get('Name', 'N/A')}")
#                     if 'HybridScore' in best_hit:
#                         st.write(f"**Hybrid Score:** {best_hit['HybridScore']:.3f}")
#                     if 'ShapeScore' in best_hit:
#                         st.write(f"**Shape Score:** {best_hit['ShapeScore']:.3f}")
#                     if 'FeatureScore' in best_hit:
#                         st.write(f"**Feature Score:** {best_hit['FeatureScore']:.3f}")

#     progress_bar.progress(100)
#     status_text.text("✅ Analysis complete!")
#     st.balloons()

# # Footer
# st.markdown("---")
# footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
# with footer_col1:
#     st.markdown("💻 *Developed with ❤️ using Streamlit + SHAFTS + RDKit + LLM reasoning (DeepSeek)*")
# with footer_col2:
#     st.markdown("[📚 Documentation](#)")
# with footer_col3:
#     st.markdown("[🐛 Report Issue](#)")