# %%
# %% read dataframe from part1
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_pickle("sqf.pkl")

# %% select some yes/no columns to convert into a dataframe of boolean values
pfs = [col for col in df.columns if col.startswith("pf_")]

stopped = [
    "cs_objcs",
    "cs_descr",
    "cs_casng",
    "cs_lkout",
    "rf_vcact",
    "cs_cloth",
    "cs_drgtr",

]

x = df[pfs + stopped]
x = x == "YES"

# %% create a new column to represent whether a person is armed
x["stopped"] = (
    x["cs_objcs"]
    | x["cs_descr"]
    | x["cs_casng"]
    | x["cs_lkout"]
    | x["rf_vcact"]
    | x["cs_cloth"]
    | x["cs_drgtr"]
)

# %% select some categorical columns and do one hot encoding
for val in df["race"].unique():
    x[f"race_{val}"] = df["race"] == val


for val in df["city"].value_counts().index:
    x[f"city_{val}"] = df["city"] == val


for val in df["sex"].value_counts().index:
    x[f"sex_{val}"] = df["sex"] == val

    # %% apply frequent itemsets mining, make sure you play around of the support level
frequent_itemsets = apriori(x, min_support=0.01, use_colnames=True)

# %% apply association rules mining
rules = association_rules(frequent_itemsets, min_threshold=0.5)
rules

# %% sort rules by confidence and select rules within "armed" in it
rules.sort_values("confidence", ascending=False)[
    rules.apply(
        lambda r: "race_WHITE" in r["antecedents"]
        or "race_WHITE" in r["consequents"],
        axis=1,
    )
]

# %%
rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules

# %% Add rules to find highest confidence
rules[ (rules['antecedent_len'] >= 2) &
       (rules['confidence'] == 1) &
       (rules['support'] > 0.3) ]

