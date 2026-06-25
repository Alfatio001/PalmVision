import pandas as pd

# ======================================================
# LOAD CSV
# ======================================================

df = pd.read_csv("tree_analysis.csv")

# ======================================================
# BASIC STATISTICS
# ======================================================

width = df["Width"]
height = df["Height"]
area = df["Area"]
canopy = df["Canopy_Size"]

print("=" * 60)
print("CANOPY STATISTICS")
print("=" * 60)

print(f"Total Trees : {len(df)}")
print()

# -----------------------------
# CANOPY SIZE
# -----------------------------
print("CANOPY SIZE")
print(f"Minimum : {canopy.min():.2f}")
print(f"Maximum : {canopy.max():.2f}")
print(f"Mean    : {canopy.mean():.2f}")
print(f"Median  : {canopy.median():.2f}")
print(f"Std Dev : {canopy.std():.2f}")

print()

# -----------------------------
# WIDTH
# -----------------------------
print("WIDTH")
print(f"Minimum : {width.min():.2f}")
print(f"Maximum : {width.max():.2f}")
print(f"Mean    : {width.mean():.2f}")
print(f"Median  : {width.median():.2f}")
print(f"Std Dev : {width.std():.2f}")

print()

# -----------------------------
# HEIGHT
# -----------------------------
print("HEIGHT")
print(f"Minimum : {height.min():.2f}")
print(f"Maximum : {height.max():.2f}")
print(f"Mean    : {height.mean():.2f}")
print(f"Median  : {height.median():.2f}")
print(f"Std Dev : {height.std():.2f}")

print()

# -----------------------------
# AREA
# -----------------------------
print("AREA")
print(f"Minimum : {area.min():.2f}")
print(f"Maximum : {area.max():.2f}")
print(f"Mean    : {area.mean():.2f}")
print(f"Median  : {area.median():.2f}")
print(f"Std Dev : {area.std():.2f}")

# ======================================================
# QUARTILES
# ======================================================

Q1 = canopy.quantile(0.25)
Q2 = canopy.quantile(0.50)
Q3 = canopy.quantile(0.75)

print()
print("=" * 60)
print("CANOPY SIZE QUARTILE")
print("=" * 60)

print(f"Q1 : {Q1:.2f}")
print(f"Q2 : {Q2:.2f}")
print(f"Q3 : {Q3:.2f}")

# ======================================================
# CLASSIFICATION
# ======================================================

def classify(canopy_size):

    if canopy_size < Q1:
        return "Small"

    elif canopy_size <= Q3:
        return "Medium"

    else:
        return "Large"

df["Canopy_Class"] = df["Canopy_Size"].apply(classify)

# ======================================================
# SUMMARY
# ======================================================

summary = df["Canopy_Class"].value_counts()

print()
print("=" * 60)
print("CANOPY DISTRIBUTION")
print("=" * 60)

print(summary)

# ======================================================
# SAVE RESULT
# ======================================================

df.to_csv("tree_analysis_final.csv", index=False)

print()
print("Saved : tree_analysis_final.csv")