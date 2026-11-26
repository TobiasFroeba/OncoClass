import json
from collections import Counter

# File paths
input_file = "./Input/OncoClass_VICC_Input.json"
output_file = "./Input/OncoClass_VICC_filter.json"

# Load JSON files
with open(input_file, "r") as f:
    input_data = json.load(f)

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

Ringversuch_IDs = [443, 86, 35, 37, 17, 27, 43, 101, 21, 413, 
                   452, 451, 450, 449, 448, 447, 446, 444, 453, 454, 
                   459, 458, 457, 456, 455, 460, 461, 462, 463, 464, 
                   13053]


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


def get_vicc_class(obj):
    score = 0

    for key, weight in VICC_FACTORS.items():
        if obj.get(key) == "Yes":
            score += weight

    return vicc_class_from_score(score)

OP4_counter = 0
OP4_entries = []
invalid_entries = []
wrong_label_entries = []

def violates_rules(obj):
    global OP4_counter
    global OP4_entries
    
    
    # Enforce OP4 criteria
    if obj.get("gnomad_af") == "None":
        if obj.get("OP4") == "No":
            OP4_entries.append(obj)
            return True
    else:
        if (obj.get("OP4") == "Yes" and obj.get("gnomad_af") >= 0.001) or (obj.get("OP4") == "No" and obj.get("gnomad_af") < 0.001):
            OP4_entries.append(obj)
            return True
    
    # Enforce official Exclusions from VICC
    if obj.get("OVS1") == "Yes" and obj.get("OM2") == "Yes":
        return True
    if obj.get("OS1") == "Yes" and obj.get("OM1") == "Yes":
        return True
    if obj.get("OS3") == "Yes" and obj.get("OM1") == "Yes":
        return True
    if obj.get("OM1") == "Yes" and obj.get("OM3") == "Yes":
        return True
    if obj.get("OM4") == "Yes" and obj.get("OM3") == "Yes":
        return True
    # Enforce inofficial OVS1/OS1 Exclusion
    if obj.get("OVS1") == "Yes" and obj.get("OS1") == "Yes":
        return True
    
    return False
    
filtered_results = []

# Loop over input entries
for entry in input_data:

    # Exclude Ringversuch IDs
    if entry.get("ID") in Ringversuch_IDs:
        print(entry.get("ID"))
        continue

    # Check if Class is an integer in (1,2,3,4,5)
    class_value = entry.get("Class")
    try:
        class_int = int(class_value)
    except (ValueError, TypeError):
        invalid_entries.append(entry)
        continue
    
    # Filter invalid Classes
    if class_int not in {1, 2, 3, 4, 5}:
        invalid_entries.append(entry)
        continue
    
    # Sanity Check for correct Criteria   
    computed_class = int(get_vicc_class(entry))
    if computed_class != class_int:
        wrong_label_entries.append(entry)
        continue
        
    # Enforce (in)official Rules
    if violates_rules(entry):
        continue

    filtered_results.append(entry)

with open(output_file, "w") as f:
    json.dump(filtered_results, f, indent=4)

print(f"Saved {len(filtered_results)} entries from {len(input_data)} to {output_file}")

# OP4
with open("./Sandbox/Wrong_OP4_entries.json", "w") as f:
    json.dump(OP4_entries, f, indent=4)
print(f"Encountered {len(OP4_entries)} wrongly assigned OP4 criteria.")
class_counts = Counter(entry.get("Class") for entry in OP4_entries)
print("Occurrences per Class:")
for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")
    
# invalid
with open("./Sandbox/invalid_entries.json", "w") as f:
    json.dump(invalid_entries, f, indent=4)
print(f"Encountered {len(invalid_entries)} invalid entries.")
class_counts = Counter(entry.get("Class") for entry in invalid_entries)
print("Occurrences per Class:")
for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")
    
# wrong
with open("./Sandbox/wrong_label_entries.json", "w") as f:
    json.dump(wrong_label_entries, f, indent=4)
print(f"Encountered {len(wrong_label_entries)} wrongly classified criteria.")
class_counts = Counter(entry.get("Class") for entry in wrong_label_entries)
print("Occurrences per Class:")
for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")