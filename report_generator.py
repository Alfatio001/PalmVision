import os
import json
import pandas as pd

# ======================================================
# LOAD DATA
# ======================================================

CSV_FILE = "tree_analysis_final.csv"

OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)

# ======================================================
# BASIC SUMMARY
# ======================================================

summary = {

    "Total Trees": int(len(df)),

    "Average Confidence": round(df["Confidence"].mean(),3),

    "Average Canopy Size": round(df["Canopy_Size"].mean(),2),

    "Average Width": round(df["Width"].mean(),2),

    "Average Height": round(df["Height"].mean(),2),

    "Average Area": round(df["Area"].mean(),2),

    "Minimum Canopy Size": round(df["Canopy_Size"].min(),2),

    "Maximum Canopy Size": round(df["Canopy_Size"].max(),2),

    "Small Canopy": int((df["Canopy_Class"]=="Small").sum()),

    "Medium Canopy": int((df["Canopy_Class"]=="Medium").sum()),

    "Large Canopy": int((df["Canopy_Class"]=="Large").sum())
}

# ======================================================
# SAVE JSON
# ======================================================

json_path = os.path.join(
    OUTPUT_DIR,
    "summary.json"
)

with open(json_path,"w") as f:

    json.dump(summary,f,indent=4)

# ======================================================
# SAVE TXT
# ======================================================

txt_path = os.path.join(
    OUTPUT_DIR,
    "summary.txt"
)

with open(txt_path,"w") as f:

    f.write("="*50+"\n")
    f.write("PALMVISION ANALYSIS REPORT\n")
    f.write("="*50+"\n\n")

    for key,value in summary.items():

        f.write(f"{key:<25}: {value}\n")

print("="*60)
print("REPORT GENERATED SUCCESSFULLY")
print("="*60)

print(f"JSON : {json_path}")
print(f"TXT  : {txt_path}")