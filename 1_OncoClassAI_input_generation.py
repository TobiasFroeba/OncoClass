import os
import re
import json
import vcfpy
import urllib.parse
from tqdm import tqdm


"""
Produces input.json (label: variant, input: DNA_change,mRNA_change, p_change, chs_cancertype)
as well as ground_truth.json (label: variant, vicc_factors, user_classification)
"""

def sanitize(input_path, output_path):
    """Fix INFO description lines by cleaning quotes and backslashes."""
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            if line.startswith("##INFO="):
                match = re.search(r'Description="(.+?)"(,|>)', line)
                if match:
                    full_desc = match.group(1)
                    cleaned = full_desc.replace('"', "'").replace("\\", "\\\\")
                    line = line.replace(
                        f'Description="{full_desc}"',
                        f'Description="{cleaned}"'
                    )
            fout.write(line)

def build_files(vcf_file, input_json):
    reader = vcfpy.Reader.from_path(vcf_file)
    total_records = reader.header.n_records if hasattr(reader.header, "n_records") else None

    input_data = []
    ground_truth_data = []

    for record in tqdm(reader, total=total_records, desc="Processing variants"):
        chrom = record.CHROM
        pos = record.POS
        ref = record.REF
        alt = record.ALT[0].value
        ID = int(record.ID[0])
        
        """
        Clearing all used variables
        """
        consequence_entry = None
        gnomad_entry = None
        alpha_missense_entry = None
        revel_entry = None
        cmc_significance_entry = None
        clinvar_entry = None
        
        """
        Split consequence field
        Format:Transcript|hgvsc|hgvsp,consequence|impact|exonnr|intronnr|genesymbol|ismaneselect|ismaneplusclinical|isensemblcanonical
        """
        consequence_entry = record.INFO.get("consequences")
        
        consequencelist = []
        genelist = []
        transcriptlist = []
        hgvsclist = []
        hgvsplist = []
        genelist = []
        
        for entry_consequence in consequence_entry:
            
            consequence_blocks = entry_consequence.split("&")
            
            for entry in consequence_blocks:

                fields = entry.split("|")

                transcript = urllib.parse.unquote(fields[0])
                hgvsc = urllib.parse.unquote(fields[1])
                hgvsp = urllib.parse.unquote(fields[2])
                consequence = urllib.parse.unquote(fields[3])
                gene = urllib.parse.unquote(fields[7])
                ismaneselect = fields[8]
                ismaneplusclinical = fields[9]
                
                
                if ismaneselect == "True":
                    transcriptlist.append(transcript)
                    hgvsclist.append(hgvsc)
                    hgvsplist.append(hgvsp)
                    consequencelist.append(consequence)
                    genelist.append(gene)

        transcripts = ",".join(dict.fromkeys(x for x in transcriptlist if x != "None"))
        hgvscs = ",".join(dict.fromkeys(x for x in hgvsclist if x != "None"))
        hgvsps = ",".join(dict.fromkeys(x for x in hgvsplist if x != "None"))
        consequences = ",".join(dict.fromkeys(x for x in consequencelist if x != "None"))
        genes = ",".join(dict.fromkeys(x for x in genelist if x != "None"))

        if record.INFO.get("gnomad_popmax_AF"):
            gnomad_entry = record.INFO.get("gnomad_popmax_AF")
        elif record.INFO.get("gnomad_af"):
            gnomad_entry = record.INFO.get("gnomad_af")
                
        try:
            gnomad = float(urllib.parse.unquote(gnomad_entry))
        except:
            gnomad = "None"
                
        alpha_missense_entry = record.INFO.get("alpha_missense")
        try:
            alpha_missense = float(alpha_missense_entry.split("$", 1)[1])
        except:
            alpha_missense = "None"

        revel_entry = record.INFO.get("revel")
        revel_scores = []
        if revel_entry:
            for entry in revel_entry.split("|"):
                try:
                    score = entry.split("$", 1)[1]
                    revel_scores.append(float(score))
                except:
                    revel = "None"
        if revel_scores:
            revel = max(revel_scores)

        cmc_significance_entry = record.INFO.get("cosmic_cmc_significance_tier")
        try:
            cmc_significance = int(cmc_significance_entry.split("$", 1)[1])
        except:
            cmc_significance = "None"

        clinvar_entry = record.INFO.get("clinvar_summary")
        clinvar_class = "None"
        if clinvar_entry:
            for line in clinvar_entry:
                try:
                    clinvar_class = urllib.parse.unquote(line.split(":", 1)[1])
                except:
                    continue
        
        """
        Read out selected criteria
        """
        factors = [
            "OVS1", "OS1", "OS2", "OS3",
            "OM1", "OM2", "OM3", "OM4",
            "OP1", "OP2", "OP3", "OP4",
            "SBVS1", "SBS1", "SBS2",
            "SBP1", "SBP2"
        ]
        factor_map = {f: "No" for f in factors}
        class_value = None

        uc = record.INFO.get("user_classification")
        if uc:
            parts = uc.split("|")
            if len(parts) > 1:
                class_value = parts[1]
            if len(parts) > 5:
                factor_blocks = parts[5].split("$")
                for block in factor_blocks:
                    for f in factors:
                        if block.startswith(f + "+") and "++selected" in block:
                            factor_map[f] = "Yes"
        
        """
        Generate JSON object
        """
        input_data.append({
            "ID": ID, 
            "chrom": chrom, 
            "pos": pos, 
            "ref": ref, 
            "alt": alt, 
            "gene": genes, 
            "transcript": transcripts, 
            "hgvsc": hgvscs, 
            "hgvsp": hgvsps if hgvsps is not None else "not documented", 
            "gnomad_af": gnomad, 
            "alpha_missense": alpha_missense, 
            "revel": revel, 
            "consequence": consequences, 
            "cmc_significance": cmc_significance, 
            "clinvar_class": clinvar_class,
            **factor_map, "Class": class_value
        })        

    with open(input_json, "w") as out1:
        json.dump(input_data, out1, indent=4)


if __name__ == "__main__":
    vcf_input = os.path.expanduser("./Input/VICC_V3.vcf")
    vcf_clean = os.path.expanduser("./Input/VICC_V3_clean.vcf")
    input_json = "./Input/OncoClass_VICC_Input.json"
    
    sanitize(vcf_input, vcf_clean)
    build_files(vcf_clean, input_json)