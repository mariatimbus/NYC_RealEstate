import pandas as pd

df = pd.read_csv("data/simplified_dataset.csv", encoding="latin1")

df = df.dropna()

numeric_cols = [
    "SALE PRICE",
    "TOTAL UNITS",
    "RESIDENTIAL UNITS",
    "COMMERCIAL UNITS",
    "GROSS SQUARE FEET",
    "LAND SQUARE FEET",
    "YEAR BUILT",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

df = df[df["SALE PRICE"] > 0]
df = df[df["TOTAL UNITS"] > 0]
df = df[df["GROSS SQUARE FEET"] > 0]
df = df[df["LAND SQUARE FEET"] > 0]
df = df[df["YEAR BUILT"] > 1800]

categorical_cols = [
    "NEIGHBORHOOD",
    "BUILDING CLASS CATEGORY",
    "BUILDING CLASS AT PRESENT",
]

for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()
    df = df[df[col] != ""]
    df = df[df[col] != "-"]

df.to_csv("data/cleaned_dataset.csv", index=False)

print("DONE")
print("Rows remaining:", len(df))