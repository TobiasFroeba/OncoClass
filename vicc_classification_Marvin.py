import json

VICC_FACTORS = {
    "OVS1": 8,
    "OS1": 4,
    "OS2": 4,
    "OS3": 4,
    "OM1": 2,
    "OM2": 2,
    "OM3": 2,
    "OM4": 2,
    "OP1": 1,
    "OP2": 1,
    "OP3": 1,
    "OP4": 1,
    "SBVS1": -8,
    "SBS1": -4,
    "SBS2": -4,
    "SBP1": -1,
    "SBP2": -1,
}

def vicc_class_from_score(score):
    if score >= 10:
        return 5
    if 6 <= score <= 9:
        return 4
    if 0 <= score <= 5:
        return 3
    if -6 <= score <= -1:
        return 2
    return 1

def get_vicc_score(obj):
    score = 0
    for key, weight in VICC_FACTORS.items():
        if obj.get(key) == "Yes" or obj.get(key) == 1:
            score += weight
    return vicc_class_from_score(score)

def fix_double_counting(obj):

    if obj.get("OS1") == "Yes" and obj.get("OM1") != "No":
        obj["OM1"] = "No"
        obj["OM1-Comment"] = "OS1 rules this factor out."
    if obj.get("OS3") == "Yes" and obj.get("OM1") != "No":
        obj["OM1"] = "No"
        obj["OM1-Comment"] = "OS3 rules this factor out."

    if obj.get("OVS1") == "Yes" and obj.get("OM2") != "No":
        obj["OM2"] = "No"
        obj["OM2-Comment"] = "OVS1 rules this factor out."

    if obj.get("OM1") == "Yes" and obj.get("OM3") != "No":
        obj["OM3"] = "No"
        obj["OM3-Comment"] = "OM1 rules this factor out."
    if obj.get("OM4") == "Yes" and obj.get("OM3") != "No":
        obj["OM3"] = "No"
        obj["OM3-Comment"] = "OM4 rules this factor out."
        
    if obj.get("OS1") == "Yes" and obj.get("OM4") != "No":
        obj["OM4"] = "No"
        obj["OM4-Comment"] = "OS1 rules this factor out."
    if obj.get("OS3") == "Yes" and obj.get("OM4") != "No":
        obj["OM4"] = "No"
        obj["OM4-Comment"] = "OS3 rules this factor out."
    if obj.get("OM1") == "Yes" and obj.get("OM4") != "No":
        obj["OM4"] = "No"
        obj["OM4-Comment"] = "OM1 rules this factor out."

"""
json_path = "./Output/OncoClass_finetune_classification_V2_run4.json"

with open(json_path) as f:
    data = json.load(f)

for obj in data:
    if not isinstance(obj["output"], dict):
        continue
    # fix_double_counting(obj["output"])
    obj["output"]["VICC_Class"] = str(get_vicc_score(obj["output"]))

with open(json_path, "w") as f:
    json.dump(data, f, indent=2)
"""