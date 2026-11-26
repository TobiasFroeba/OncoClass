import os
import json
import random
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

input_df = pd.read_json("./Input/OncoClass_VICC_filter.json")

N = 3700
sampled_frames = []

for cls, group in input_df.groupby("Class"):
    if len(group) <= N:
        sampled = group
    else:
        sampled = group.sample(N, random_state=42)
    sampled_frames.append(sampled)

if sampled_frames:
    sampled_df = pd.concat(sampled_frames, axis=0).reset_index(drop=True)
    print(sampled_df["Class"].value_counts().sort_index())

# split in preparation for finetuning
sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_df, test_df = train_test_split(
    sampled_df,
    test_size=0.3,
    random_state=42,
    stratify=sampled_df["Class"]
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print("Training set:")
print(train_df["Class"].value_counts().sort_index())

print("\nTest set:")
print(test_df["Class"].value_counts().sort_index())

# Compare relative class distributions
print("\nClass proportions:")
print(pd.DataFrame({
    "Original": round(sampled_df["Class"].value_counts(normalize=True), 2),
    "Train": round(train_df["Class"].value_counts(normalize=True), 2),
    "Test": round(test_df["Class"].value_counts(normalize=True), 2)
}).sort_index())

print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")
test_df.to_json("./Input/OncoClass_test.json", orient="records", indent=2)

### Prepare Train Set ###

df_ncg = pd.read_csv("./RAG/NCG_cancerdrivers_annotation_supporting_evidence.tsv", sep="\t")
df_okb = pd.read_csv("./RAG/OncoKB_cancerGeneList.tsv", sep="\t")

input_data = train_df.to_dict(orient="records")

print(f"Input size: {len(input_data)}.")

with open("./Prompt/System_Instruction.txt", "r", encoding="utf-8") as f:
    system_instruction = f.read()
    
combined = []

def syntheticamrevel(ref, alt, revel, alpha_missense, consequencelist):
    # Synthetic REVEL & AlphaMissense values
    high = {"splice_acceptor_variant","splice_donor_variant","stop_gained","stop_lost","start_lost"}
    low = {"start_retained_variant","stop_retained_variant","synonymous_variant"}
    
    if len(ref) == len(alt):
        if revel is None:
            if any(x in consequencelist for x in high):
                revel = 0.99
            elif any(x in consequencelist for x in low):
                revel = 0
        if alpha_missense is None:
            if any(x in consequencelist for x in high):
                alpha_missense = 0.99
            elif any(x in consequencelist for x in low):
                alpha_missense = 0
    return revel, alpha_missense
    
    
def tsgoncogene(genelist, df_ncg, df_okb):
    # RAG: get tsg/oncogene info from NCG and OncoKB
    tsg_ncg = oncogene_ncg = tsg_okb = oncogene_okb = False
    for gene in genelist.split(","):
        ncg_rag_matches = list(df_ncg.loc[df_ncg["symbol"] == gene, ["NCG_oncogene", "NCG_tsg"]].itertuples(index=False, name=None))
        ncg_rag_matches = [(None if pd.isna(o) else o, None if pd.isna(t) else t) for o, t in ncg_rag_matches]
        if any(t == 1 for _, t in ncg_rag_matches):
            tsg_ncg = True
        if any(o == 1 for o, _ in ncg_rag_matches):
            oncogene_ncg = True
    
        oncoKB_rag_matches = df_okb.loc[(df_okb["Hugo Symbol"] == gene) | (df_okb["Gene Aliases"].str.contains(fr'(?:^|,\s*){gene}(?:,|$)', na=False)), "Gene Type"]
        if "TSG" in oncoKB_rag_matches.values or "ONCOGENE_AND_TSG" in oncoKB_rag_matches.values:
            tsg_okb = True
        if "ONCOGENE" in oncoKB_rag_matches.values or "ONCOGENE_AND_TSG" in oncoKB_rag_matches.values:
            oncogene_okb = True
        
    tsg = tsg_ncg or tsg_okb
    oncogene = oncogene_ncg or oncogene_okb
    return tsg,oncogene
    
for entry in tqdm(input_data, desc="Generating finetuning dataset"):

    unique_id = entry.get("ID")
    chrom = entry["chrom"]
    pos = entry["pos"]
    ref = entry["ref"]
    alt = entry["alt"]
    genelist = entry["gene"]
    hgvsclist = entry["hgvsc"]
    hgvsplist = entry["hgvsp"]
    gnomad = entry["gnomad_af"] if isinstance(entry["gnomad_af"], float) else None
    alpha_missense = entry["alpha_missense"] if isinstance(entry["alpha_missense"], float) else None
    revel = entry["revel"] if isinstance(entry["revel"], float) else None
    cmc_significance = entry["cmc_significance"]
    clinvar_class = entry["clinvar_class"]
    consequencelist = entry["consequence"]
        
    # Synthetic REVEL & AlphaMissense values
    revel,alphamissense = syntheticamrevel(ref, alt, revel, alpha_missense, consequencelist)
    
    # RAG: get tsg/oncogene info from NCG and OncoKB
    tsg,oncogene = tsgoncogene(genelist, df_ncg, df_okb)
    
    input_prompt = (
f"The mutation occurs on chromosome {chrom} at position {pos} where the original base {ref} is replaced by the base {alt}."
f" This affects the gene(s) {genelist}"
f"{', which is/contains a known tumor suppressor gene.' if tsg 
else ', which is/contains a known oncogene.' if oncogene 
else '.'}"
f" The variant is classified as a {consequencelist}"
f"{', which marks the variants as a null mutation.' if any(v in consequencelist for v in ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost'])
else ', which results in protein length changes.' if any(v in consequencelist for v in ['inframe_insertion', 'inframe_deletion'])
else '.'}"
f" The variant is classified as {clinvar_class} in ClinVar."
f" The COSMIC cmc significance is {cmc_significance}."

f"{' The variant is therefore oncogenic.' if any(v in clinvar_class for v in ['Pathogenic', 'Likely_pathogenic']) or cmc_significance in (1, 2, 3) 
else ''}"
f"{' Functional studies show no oncogenic effect for this variant.' if any(v in clinvar_class for v in ['Benign', 'Likely_benign']) 
else ''}"

f"{' Computational evidence supports an oncogenic effect for this variant.' if revel is not None and revel > 0.5 and alpha_missense is not None and alpha_missense > 0.56 
else ' Computational evidence shows no oncogenic evidence for this variant.' if revel is not None and revel <= 0.5 and alpha_missense is not None and alpha_missense < 0.34
else ''}"

f"{' This variant is absent from controls in gnomAD.' if gnomad is None or gnomad < 0.001
else ' This variant exhibits a medium minor allele frequency in gnomAD (' + str(round(gnomad, 3)) + ').' if gnomad is not None and 0.01 < gnomad <= 0.05
else ' This variant exhibits a high minor allele frequency in gnomAD (' + str(round(gnomad, 3)) + ').' if gnomad is not None and gnomad > 0.05
else ' The gnomAD allele frequency is ' + str(round(gnomad, 3)) + '.'}"
    )

    output_str = ", ".join(f"{k}: {v}" for k, v in entry.items() if k in {
            "OVS1", 
            "OS1", "OS2", "OS3", 
            "OM1", "OM2", "OM3", "OM4", 
            "OP1", "OP2", "OP3", "OP4", 
            "SBVS1", 
            "SBS1", "SBS2", 
            "SBP1", "SBP2",
            "Class",
    })
    combined.append({
            "input": f"{input_prompt}".replace("\n", " "),
            "output": output_str
    })

random.seed(42)
random.shuffle(combined)
n = len(combined)
n_train = int(n * 0.9)
n_val = n - n_train
train = combined[:n_train]
val = combined[n_train:n]

def write_jsonl(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

train_file = os.path.expanduser("./Input/OncoClass_train_final.json")
val_file = os.path.expanduser("./Input/OncoClass_val_final.json")

write_jsonl(train_file, train)
write_jsonl(val_file, val)

print(f"Train size: {len(train)}")
print(f"Validation size: {len(val)}")