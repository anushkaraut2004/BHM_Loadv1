#!/usr/bin/env python3
"""
merge_crossings_with_load.py
Merge vehicle_crossings.csv with vehicle_load_lookup.csv to produce
vehicle_crossings_with_load.csv where each crossing row has the standard weight.
"""

import os
import sys
import pandas as pd

# ---------- Files (edit if needed) ----------
CROSSINGS_FILE = "vehicle_crossings.csv"
LOOKUP_FILE    = "vehicle_load_lookup.csv"
OUTPUT_FILE    = "vehicle_crossings_with_load.csv"

# ---------- Safety checks ----------
for p in (CROSSINGS_FILE, LOOKUP_FILE):
    if not os.path.exists(p):
        print(f"ERROR: required file not found: {p}")
        print("Make sure both CSVs are in the same folder as this script.")
        sys.exit(1)

# ---------- Read CSVs ----------
# Read crossings as text to preserve timestamp formatting
crossings = pd.read_csv(CROSSINGS_FILE, dtype=str, keep_default_na=False)
lookup    = pd.read_csv(LOOKUP_FILE, dtype=str, keep_default_na=False)

# ---------- Basic cleaning ----------
# drop rows that are completely empty
crossings = crossings.dropna(how="all")
lookup    = lookup.dropna(how="all")

# ensure expected column names exist
if "vehicle" not in crossings.columns:
    # try to guess common alternative column names
    alt = [c for c in crossings.columns if "veh" in c.lower()]
    if alt:
        print(f"Notice: using column '{alt[0]}' as vehicle column.")
        crossings = crossings.rename(columns={alt[0]: "vehicle"})
    else:
        print("ERROR: 'vehicle' column not found in vehicle_crossings.csv")
        print("Columns found:", list(crossings.columns))
        sys.exit(1)

if "vehicle_class" not in lookup.columns:
    # try to find a reasonable column
    alt = [c for c in lookup.columns if "class" in c.lower() or "vehicle" in c.lower()]
    if alt:
        print(f"Notice: using column '{alt[0]}' as vehicle_class in lookup.")
        lookup = lookup.rename(columns={alt[0]: "vehicle_class"})
    else:
        print("ERROR: 'vehicle_class' column not found in vehicle_load_lookup.csv")
        print("Columns found:", list(lookup.columns))
        sys.exit(1)

if "standard_weight_kg" not in lookup.columns:
    alt = [c for c in lookup.columns if "weight" in c.lower() or "kg" in c.lower()]
    if alt:
        print(f"Notice: using column '{alt[0]}' as standard_weight_kg in lookup.")
        lookup = lookup.rename(columns={alt[0]: "standard_weight_kg"})
    else:
        print("ERROR: 'standard_weight_kg' column not found in vehicle_load_lookup.csv")
        print("Columns found:", list(lookup.columns))
        sys.exit(1)

# strip whitespace and normalize vehicle names to lowercase for reliable matching
crossings["vehicle"] = crossings["vehicle"].astype(str).str.strip().str.lower()
lookup["vehicle_class"] = lookup["vehicle_class"].astype(str).str.strip().str.lower()
lookup["standard_weight_kg"] = lookup["standard_weight_kg"].astype(str).str.strip()

# drop rows in crossings where vehicle is empty string after cleaning
mask_valid_vehicle = crossings["vehicle"].astype(bool)  # False for "" or " "
crossings = crossings[mask_valid_vehicle].copy()

# ---------- Merge (left join) ----------
merged = crossings.merge(
    lookup,
    left_on="vehicle",
    right_on="vehicle_class",
    how="left",
    validate="m:1"  # many crossings can match one lookup row
)

# remove the extra vehicle_class column (we already have 'vehicle')
if "vehicle_class" in merged.columns:
    merged = merged.drop(columns=["vehicle_class"])

# Attempt to coerce weight column to numeric (so missing become NaN)
merged["standard_weight_kg"] = pd.to_numeric(merged["standard_weight_kg"], errors="coerce")

# ---------- Report & save ----------
total_rows = len(crossings)
merged_rows = len(merged)
missing_weights = merged["standard_weight_kg"].isna().sum()
missing_vehicles = sorted(set(merged.loc[merged["standard_weight_kg"].isna(), "vehicle"].tolist()))

print(f"Input crossings rows : {total_rows}")
print(f"Output merged rows   : {merged_rows}")
print(f"Rows missing weight  : {missing_weights}")

if missing_weights > 0:
    print("Missing lookup for vehicle classes (sample up to 10):")
    print(missing_vehicles[:10])
    print("If you want, add these classes to vehicle_load_lookup.csv")

# Reorder columns: keep common ordering if present
desired_front = ["timestamp", "vehicle", "track_id", "frame"]
final_cols = [c for c in desired_front if c in merged.columns] + \
             [c for c in merged.columns if c not in desired_front + ["standard_weight_kg"]] + \
             (["standard_weight_kg"] if "standard_weight_kg" in merged.columns else [])

# ensure unique
final_cols = []
seen = set()
for c in merged.columns:
    if c not in seen:
        final_cols.append(c)
        seen.add(c)
# but place weight at end
if "standard_weight_kg" in final_cols:
    final_cols.remove("standard_weight_kg")
    final_cols.append("standard_weight_kg")

merged = merged[final_cols]

# Save
merged.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved -> {OUTPUT_FILE}")
print("\nPreview (first 10 rows):")
print(merged.head(10).to_string(index=False))