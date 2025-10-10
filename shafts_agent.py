# === Cynthia-specific wrapper & helpers ===
import os, shutil, subprocess, glob, json, re
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolAlign, AllChem
from rdkit.Chem.Draw import IPythonConsole
from PIL import Image
import pandas as pd
from openai import OpenAI
import requests
from typing import Optional, Dict, Any, List

def find_executable(name_or_path: str) -> str:
    if os.path.exists(name_or_path) and os.access(name_or_path, os.X_OK):
        return os.path.abspath(name_or_path)
    p = shutil.which(name_or_path)
    if p:
        return p
    raise FileNotFoundError(f"Executable '{name_or_path}' not found in PATH and path does not exist.")

def guess_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    if ext in ("sdf","sd"): return "sdf"
    if ext in ("mol2","ml2"): return "mol2"
    if ext in ("mol",): return "mol"
    if ext in ("pdb",): return "pdb"
    if ext in ("smi","smiles"): return "smi"
    return ext or "sdf"

def convert_with_obabel(in_path: str, out_path: str) -> str:
    """
    Convert in_path -> out_path using obabel. Requires 'obabel' on PATH.
    Returns out_path. Raises if obabel missing or conversion fails.
    """
    ob = shutil.which("obabel")
    if not ob:
        raise FileNotFoundError("Open Babel 'obabel' (or 'obabel') not found on PATH. Install openbabel or provide mol2 files.")
    in_fmt = guess_format(in_path)
    out_fmt = guess_format(out_path)
    # Use list args to avoid shell quoting issues
    cmd = [ob, "-i", in_fmt, in_path, "-o", out_fmt, "-O", out_path, "--errorlevel", "0"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"obabel conversion failed. stdout:{proc.stdout}\nstderr:{proc.stderr}")
    if not os.path.exists(out_path):
        raise RuntimeError(f"obabel reported success but {out_path} doesn't exist.")
    return out_path

def run_cynthia(query_path: str,
                target_path: str,
                cynthia_exe: str = "Cynthia",
                out_prefix: Optional[str] = None,
                n_hits: Optional[int] = None,
                sCutoff: Optional[float] = None,
                postOpt: bool = False,
                volumeMode: Optional[str] = None,   # "cheap" or "expensive"
                filter_file: Optional[str] = None,
                maxFeatures: Optional[int] = None,
                mtemplates: Optional[str] = None,   # path to multiple templates file
                suppressOutput: bool = False,
                weightFactor: Optional[float] = None,
                scoreOnly: bool = False,
                normalizeType: Optional[str] = None,  # dice, cosine, tanimoto, query, target
                timeout: int = 600) -> Dict[str, Any]:
    """
    Run Cynthia (SHAFTS) with the CLI options you supplied.
    Ensures inputs are MOL2 (converts using obabel if necessary).
    Returns dict: {returncode, stdout, stderr, log_path, produced_files:list}
    """
    exe_path = find_executable(cynthia_exe)
    # ensure out_prefix
    if out_prefix is None:
        out_prefix = os.path.join(os.getcwd(), "cynthia_out")
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    # Prepare mol2 inputs (Cynthia -q and -t expect .mol2)
    q_mol2 = query_path
    t_mol2 = target_path
    if guess_format(query_path) != "mol2":
        q_mol2 = os.path.splitext(query_path)[0] + ".mol2"
        convert_with_obabel(query_path, q_mol2)
    if guess_format(target_path) != "mol2":
        t_mol2 = os.path.splitext(target_path)[0] + ".mol2"
        convert_with_obabel(target_path, t_mol2)

    # Build argument list (list-style to avoid shell=True)
    args = [exe_path, "-q", q_mol2, "-t", t_mol2]
    if n_hits is not None:
        args += ["-n", str(int(n_hits))]
    if out_prefix:
        args += ["-o", out_prefix]
    if sCutoff is not None:
        args += ["-sCutoff", str(float(sCutoff))]
    if postOpt:
        args += ["-postOpt"]
    if volumeMode is not None:
        if volumeMode not in ("cheap","expensive"):
            raise ValueError("volumeMode must be 'cheap' or 'expensive'")
        args += ["-volumeMode", volumeMode]
    if filter_file:
        args += ["-filter", filter_file]
    if maxFeatures is not None:
        args += ["-maxFeatures", str(int(maxFeatures))]
    if mtemplates:
        args += ["-mtemplates", mtemplates]
    if suppressOutput:
        args += ["-suppressOutput"]
    if weightFactor is not None:
        args += ["-weightFactor", str(float(weightFactor))]
    if scoreOnly:
        args += ["-scoreOnly"]
    if normalizeType is not None:
        args += ["-normalizeType", normalizeType]

    # Keep log path
    log_path = out_prefix + ".log"
    # Run
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    # write log
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("COMMAND: " + " ".join(args) + "\n\n")
        fh.write("STDOUT:\n")
        fh.write(proc.stdout + "\n\n")
        fh.write("STDERR:\n")
        fh.write(proc.stderr + "\n")
    # collect produced files that have the prefix
    produced = sorted(glob.glob(out_prefix + "*"))
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "log_path": log_path,
        "produced_files": produced,
        "cmd": args
    }

def parse_cynthia_output(output_prefix: str):
    """
    Parse the SHAFTS (Cynthia) output file into a structured format.

    Parameters
    ----------
    output_prefix : str
        The prefix used when running Cynthia (e.g., "results/aspirin_vs_db").
        The corresponding output file will be "<output_prefix>Result.list".

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns: Rank, Name, HybridScore, ShapeScore, FeatureScore, Query
    """
    output_file = f"{output_prefix}Result.list"

    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Expected output file not found: {output_file}")

    with open(output_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Find header line (should contain column names)
    header_idx = None
    for i, line in enumerate(lines):
        if line.lower().startswith("rank"):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Header line not found in Cynthia output file.")

    # Read table section
    header = lines[header_idx].split('\t')
    data_lines = lines[header_idx + 1:]

    parsed_data = []
    for line in data_lines:
        parts = line.split('\t')
        if len(parts) != len(header):
            # Skip malformed lines
            continue
        record = dict(zip(header, parts))
        # Convert numeric columns
        for col in ['Rank', 'HybridScore', 'ShapeScore', 'FeatureScore']:
            if col in record:
                try:
                    record[col] = float(record[col]) if col != 'Rank' else int(record[col])
                except ValueError:
                    pass
        parsed_data.append(record)

    if not parsed_data:
        raise ValueError("No valid data parsed from Cynthia output.")

    results_df = pd.DataFrame(parsed_data)
    results_df = results_df.drop(columns=['Query'])
    return results_df

def visualize_best_hit(results_df: pd.DataFrame, db_file: str, top_n: int = 1, save_path: str = None):
    """
    Visualize the top-N hits from Cynthia (SHAFTS) results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Parsed results from Cynthia output.
    db_file : str
        Path to SDF or MOL2 file containing target molecules.
    top_n : int
        Number of top hits to visualize (default = 1).
    save_path : str, optional
        If provided, save the visualization image to this path.
    """
    # Sort by HybridScore descending
    top_hits = results_df.sort_values(by="HybridScore", ascending=False).head(top_n)
    
    print(f"\nTop {top_n} hit(s):")
    print(top_hits[['Rank', 'Name', 'HybridScore', 'ShapeScore', 'FeatureScore']])
    
    # hit_names = set(top_hits['Name'].tolist())
    hit_order = top_hits['Name'].tolist()

    # Detect file format
    if db_file.lower().endswith(".sdf"):
        suppl = Chem.SDMolSupplier(db_file, removeHs=False)
    elif db_file.lower().endswith(".mol2"):
        suppl = Chem.Mol2MolSupplier(db_file, removeHs=False)
    else:
        raise ValueError("Unsupported file format. Must be .sdf or .mol2")

    mol_dict = {}
    for mol in suppl:
        if mol is None:
            continue
        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else None
        if mol_name in hit_order:
            mol_dict[mol_name] = mol
    matched_mols = [(name, mol_dict[name]) for name in hit_order if name in mol_dict]
    
    if not matched_mols:
        raise ValueError("No matching molecules found in database file.")

    print(f"\n✅ Found {len(matched_mols)} matching molecule(s) in the DB file.")
    
    # Draw molecules (2D projection)
    mols = [m for _, m in matched_mols]
    legends = [n for n, _ in matched_mols]
    img = Draw.MolsToGridImage(mols, legends=legends, molsPerRow=top_n, subImgSize=(300, 300), returnPNG=False)
    if save_path:
        img.save(save_path)
        print(f"💾 Visualization saved to {save_path}")
    
    return img

client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")

# ------------------------------------------------------------
# Utility: Extract structured variables from user input
# ------------------------------------------------------------
def extract_query_info(user_prompt: str):
    """
    Extract structured info from a user natural language command.
    Example:
    "Now compare similarity between O=C(C)Oc1ccccc1C(=O)O and our mini lab compounds. Show top 5 hits."
    """
    system_prompt = """You are a chemistry-oriented LLM agent. 
    Your task is to extract structured information for molecular similarity search.
    Return a JSON with these fields:
    - query_smiles: str (SMILES string if present)
    - query_name: str (compound name, optional)
    - num_shown: int (top N hits to show, default 10 if not specified)
    Make sure the output is a valid JSON only, with no explanation.
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    raw_output = response.choices[0].message.content.strip()

    # Attempt to extract valid JSON
    try:
        data = json.loads(raw_output)
    except json.JSONDecodeError:
        # fallback: try to clean up any trailing text
        match = re.search(r"\{.*\}", raw_output, re.S)
        data = json.loads(match.group(0)) if match else {}

    # Default num_shown = 10
    if "num_shown" not in data:
        data["num_shown"] = 10

    return data

# ------------------------------------------------------------
# PubChem lookup: SMILES <-> Name
# ------------------------------------------------------------
def get_compound_name_from_smiles(smiles: str) -> str:
    """
    Search PubChem for the given SMILES and return its common name.
    If not found, return 'query'.
    """
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/JSON"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        name = data["PC_Compounds"][0]["props"][0]["value"]["sval"]
        return name if name else "query"
    except Exception:
        return "query"

def get_smiles_from_name(name: str) -> str:
    """
    Try to resolve a chemical name to SMILES.
    1) Try PubChem REST API.
    2) If not found, use LLM reasoning to infer approximate SMILES.
    """
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        smiles = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        return smiles
    except Exception:
        # fallback to LLM reasoning
        print(f"⚠️ PubChem lookup failed for '{name}', using LLM reasoning...")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a chemistry model. Given a compound name, output its most likely canonical SMILES string. Return only SMILES, no explanations."
                },
                {"role": "user", "content": f"Compound name: {name}"}
            ],
            temperature=0
        )
        raw_smiles = response.choices[0].message.content.strip()
        # basic validation
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol:
            return raw_smiles
        else:
            raise ValueError(f"Failed to infer valid SMILES for {name}. Got: {raw_smiles}")

# ------------------------------------------------------------
# Molecule file preparation
# ------------------------------------------------------------  
def smiles_to_sdf(smiles: str, compound_name: str):
    """
    Generate 3D-optimized molecule with hydrogens using RDKit.
    Exports both SDF and MOL2 for SHAFTS/Cynthia.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    mol.SetProp("_Name", compound_name)

    sdf_path = f"{compound_name}.sdf"
    mol2_path = f"{compound_name}.mol2"

    # Save SDF
    w = Chem.SDWriter(sdf_path)
    w.write(mol)
    w.close()

    # # Save MOL2 (via RDKit’s MolToMol2File)
    # from rdkit.Chem import rdmolfiles
    # rdmolfiles.MolToMol2File(mol, mol2_path)

    return sdf_path

def run_llm_driven_similarity(user_prompt: str):
    # Step 1: LLM reasoning
    info = extract_query_info(user_prompt)
    print("LLM Extracted Info:", info)

    query_smiles = info.get("query_smiles", "").strip()
    query_name = info.get("query_name", "").strip()
    num_shown = info.get("num_shown", 10)
    
    if not query_smiles and not query_name:
        raise ValueError("Neither SMILES nor compound name provided.")
    
    if query_smiles and not query_name:
        query_name = get_compound_name_from_smiles(query_smiles)
    # print(f"Resolved compound name: {query_name}")
    
    if query_name and not query_smiles:
        query_smiles = get_smiles_from_name(query_name)
    print(f"Resolved molecule: {query_name} | {query_smiles}")

    # Step 2: Save query as SDF/mol2 for Cynthia
    query_sdf = smiles_to_sdf(query_smiles, query_name)
    print(f"Generated {query_sdf}")
    # query_sdf = f"{query_name}.sdf"
    query_mol2 = f"{query_name}.mol2"

    # Convert SMILES → mol2 using obabel
    import subprocess
    # subprocess.run(["obabel", "-:" + query_smiles, "-O", query_sdf])
    subprocess.run(["obabel", query_sdf, "-O", query_mol2])

    # Step 3: Run Cynthia
    res = run_cynthia(
        query_path=query_mol2,
        target_path="random_mini_db.sdf",
        cynthia_exe="Cynthia",
        out_prefix=f"results\\{query_name}_vs_db",
        timeout=100
    )

    print("Cynthia rc:", res["returncode"])
    print("Produced files:", res["produced_files"])
    print("Log file:", res["log_path"])

    # Step 4: Parse output
    results = parse_cynthia_output(f"results\\{query_name}_vs_db")
    print(results.head(num_shown))

    # Step 5 (optional): Visualize best hit
    img = visualize_best_hit(results, db_file="random_mini_db.sdf", top_n=num_shown, save_path="best_hit.png")

    return results.head(num_shown)

def interactive_agent():
    print("🧬 SHAFTS-Agent Interactive Mode (powered by DeepSeek + RDKit + Cynthia)")
    print("Type your prompt (e.g. 'Compare similarity between aspirin and our mini lab compounds. Show top 5 hits.')")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_prompt = input(">>> ")
        if user_prompt.lower().strip() in ["exit", "quit"]:
            print("👋 Exiting SHAFTS-Agent. Bye!")
            break
        try:
            run_llm_driven_similarity(user_prompt)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Example usage
    # res = run_cynthia(
    #     query_path="query.sdf",         # will be converted to query.mol2 via obabel if needed
    #     target_path="my_mini_db.sdf",          # will be converted to db.mol2
    #     cynthia_exe="Cynthia", # or just "Cynthia" if on PATH
    #     out_prefix="results\\aspirin_vs_db",
    #     # n_hits=10,
    #     # sCutoff=0.2,
    #     # postOpt=True,
    #     # volumeMode="expensive",
    #     # maxFeatures=60,
    #     # suppressOutput=False,
    #     # weightFactor=0.6,
    #     # normalizeType="dice",
    #     timeout=100
    # )
    # print("Cynthia rc:", res["returncode"])
    # print("Produced files:", res["produced_files"])
    # print("Log file:", res["log_path"])

    # # parse textual stdout/log for matches
    # output_prefix = "results\\aspirin_vs_db"
    # results = parse_cynthia_output(output_prefix)
    # num_shown = 10
    # print(results.head(num_shown))
    
    # img = visualize_best_hit(results, db_file="my_mini_db.sdf", top_n=3, save_path="best_hit.png")
    
    # user_prompt = "Now compare similarity between aspirin, and our mini lab compounds. Show top 5 hits."
    # top_results = run_llm_driven_similarity(user_prompt)
    interactive_agent()