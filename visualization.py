import os
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# LOAD DATA
# ======================================================

CSV_FILE = "tree_analysis_final.csv"

OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)

# ======================================================
# PIE CHART
# ======================================================

plt.figure(figsize=(6,6))

df["Canopy_Class"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.ylabel("")
plt.title("Canopy Class Distribution")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "canopy_distribution.png"
    )
)

plt.close()

# ======================================================
# CANOPY HISTOGRAM
# ======================================================

plt.figure(figsize=(8,5))

plt.hist(
    df["Canopy_Size"],
    bins=30
)

plt.title("Canopy Size Distribution")
plt.xlabel("Canopy Size (Pixel)")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "canopy_histogram.png"
    )
)

plt.close()

# ======================================================
# CONFIDENCE HISTOGRAM
# ======================================================

plt.figure(figsize=(8,5))

plt.hist(
    df["Confidence"],
    bins=20
)

plt.title("Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confidence_histogram.png"
    )
)

plt.close()

# ======================================================
# WIDTH HISTOGRAM
# ======================================================

plt.figure(figsize=(8,5))

plt.hist(
    df["Width"],
    bins=30
)

plt.title("Bounding Box Width")
plt.xlabel("Pixel")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "width_histogram.png"
    )
)

plt.close()

# ======================================================
# HEIGHT HISTOGRAM
# ======================================================

plt.figure(figsize=(8,5))

plt.hist(
    df["Height"],
    bins=30
)

plt.title("Bounding Box Height")
plt.xlabel("Pixel")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "height_histogram.png"
    )
)

plt.close()

# ======================================================
# AREA HISTOGRAM
# ======================================================

plt.figure(figsize=(8,5))

plt.hist(
    df["Area"],
    bins=30
)

plt.title("Bounding Box Area")
plt.xlabel("Pixel²")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "area_histogram.png"
    )
)

plt.close()

# ======================================================
# BAR CHART
# ======================================================

plt.figure(figsize=(7,5))

df["Canopy_Class"].value_counts().plot(kind="bar")

plt.title("Number of Trees by Canopy Class")
plt.xlabel("Class")
plt.ylabel("Number of Trees")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "canopy_bar_chart.png"
    )
)

plt.close()

print("="*60)
print("Visualization Completed")
print("="*60)

print(f"Graphs saved in : {OUTPUT_DIR}")