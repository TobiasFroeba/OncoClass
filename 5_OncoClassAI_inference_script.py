import os
import re
import json
import math
import time
import requests
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import vicc_classification_Marvin

"""
Assumes server hosting llama-3.1-8b-instruct is running at port 127.0.0.1:8002
Sends static prompt + single line from individual alteration json file.
Returns json object with VICC factors & cancer classification.
"""

with open("./Prompt/JSON_Schema.json", "r", encoding="utf-8") as f:
    json_schema = json.load(f)

with open("./Prompt/System_Instruction_OncoClass.txt", "r", encoding="utf-8") as f:
    system_instruction = f.read()


def prompt(unique_id, chrom, pos, ref, alt, genelist, hgvsclist, hgvsplist, tsg, oncogene, gnomad, alpha_missense, revel, consequencelist, cmc_significance, clinvar_class):
    
    user_content = (
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
else ' This variant exhibits a medium minor allele frequency in gnomAD (' + str(gnomad) + ').' if gnomad is not None and 0.01 < gnomad <= 0.05
else ' This variant exhibits a high minor allele frequency in gnomAD (' + str(gnomad) + ').' if gnomad is not None and gnomad > 0.05
else ' The gnomAD allele frequency is ' + str(gnomad) + '.'}"
    )

    url = "http://127.0.0.1:8001/v1/chat/completions"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "OncoClass_Mistral_LoRA_Adapter",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": 1000,
        "temperature": 0.1,
        "structured_outputs": {
            "json": json_schema
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    content = response.json()
    # print(content)
    output = content["choices"][0]["message"]["content"].strip()
    output_dict = json.loads(output)
    
    # print(user_content)
    # print(output_dict)
    
    """
    # Automate OVS1
    # null mutation in tumor suppressor gene
    if tsg and any(v in consequencelist for v in ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost']):
        output_dict["OVS1"] = "Yes"
    else:
        output_dict["OVS1"] = "No"
     
    # Automate OS1
    # exclusion rule with OVS1, enhancing mutation in oncogene
    if output_dict["OVS1"] == "Yes":
        output_dict["OS1"] = "No"
        output_dict["OS1-Comment"] = "OVS1 rules this factor out."
    elif oncogene and (cmc_significance in (1,2,3) or any(v in clinvar_class for v in ['Pathogenic', 'Likely_pathogenic'])) and not any(v in consequencelist for v in ['stop_gained', 'frameshift_variant', 'splice_acceptor_variant', 'splice_donor_variant', 'start_lost']):
        output_dict["OS1"] = "Yes"
    else:
        output_dict["OS1"] = "No"
    """
    
    return {
    "ID": unique_id, 
    "output": output_dict
    }
    
    
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

if __name__ == "__main__":
    # Prepare input
    input_file = os.path.expanduser("./Input/OncoClass_test.json")
    with open(input_file, "r") as f:
        input_data = json.load(f)
        
    df_ncg = pd.read_csv("./RAG/NCG_cancerdrivers_annotation_supporting_evidence.tsv", sep="\t")
    df_okb = pd.read_csv("./RAG/OncoKB_cancerGeneList.tsv", sep="\t")
    
    results = []
    retry = []
    missed = 0
    
    for entry in tqdm(input_data, desc="Processing mutations"):
        unique_id = entry["ID"]
        chrom = entry["chrom"]
        pos = entry["pos"]
        ref = entry["ref"]
        alt = entry["alt"]
        genelist = entry["gene"]
        hgvsclist = entry["hgvsc"].split(",")
        hgvsplist = entry["hgvsp"].split(",")
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
        
        # Submit query
        for n in range(1,21):
            try:
                response = prompt(unique_id, chrom, pos, ref, alt, genelist, hgvsclist, hgvsplist, tsg, oncogene, gnomad, alpha_missense, revel, consequencelist, cmc_significance, clinvar_class)
                results.append(response)
                break
                
            except Exception as e:
                print(f"\n Attempt Nr. {n}")
                if n == 20:
                    missed += 1
                    retry.append(unique_id)
                    print("\n All attempts failed")
            
    print(f"{missed} variants ({round((missed/len(input_data)), 2)*100}%) could not be classified after too many failed attempts.")
    with open("./Output/MissedVariants.txt", "w") as f:
        f.write("\n".join(map(str, retry)))
        
    for obj in results:
        if not isinstance(obj["output"], dict):
            continue
            
        # Automate doublecounting-exclusion
        vicc_classification_Marvin.fix_double_counting(obj["output"])
        
        # Compute VICC-Class from criteria weights
        obj["VICC_Class"] = str(vicc_classification_Marvin.get_vicc_score(obj["output"]))
        
    df = pd.DataFrame(results)
    df.to_json("./Output/OncoClass_Mistral_Finetune_final.json", orient="records", indent=4)
    predictions = df.to_dict(orient="records")
