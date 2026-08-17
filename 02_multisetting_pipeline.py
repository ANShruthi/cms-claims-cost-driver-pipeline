"""
Extended CMS DE-SynPUF pipeline: combines INPATIENT + OUTPATIENT claims
into a multi-setting episode cost measure, matching the original
Sagar/Lin/Castel (2017) approach more closely than the inpatient-only
version. Cancer type is identified from ICD-9 codes appearing on
EITHER claim type, increasing cohort yield.

Data: CMS 2008-2010 DE-SynPUF, Sample 1 (official CMS synthetic
Medicare data -- real file structure and real ICD-9 codes, synthetic
beneficiary-level values).
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

pd.set_option("display.width", 160)
RAW_DATA_DIR = "./data/raw"
OUTPUT_DIR = "./data"

DGNS_COLS = [f"ICD9_DGNS_CD_{i}" for i in range(1, 11)]

def classify_cancer_type(row):
    codes = [str(row[c]) for c in DGNS_COLS if pd.notna(row.get(c))]
    for code in codes:
        c3 = code[:3]
        if c3 in ("174", "175"):
            return "Breast"
        if c3 == "162":
            return "Lung"
        if c3 in ("153", "154"):
            return "Colorectal"
    return None

# ---------------------------------------------------------------
# 1. Beneficiary summary + comorbidity flags (same as before)
# ---------------------------------------------------------------
bene = pd.read_csv(f"{RAW_DATA_DIR}/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv")
chronic_cols = ["SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
                 "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS",
                 "SP_RA_OA", "SP_STRKETIA"]
bene["comorbidity_count"] = (bene[chronic_cols] == 1).sum(axis=1)
cancer_ids = set(bene.loc[bene["SP_CNCR"] == 1, "DESYNPUF_ID"])
print(f"Cancer-flagged beneficiaries: {len(cancer_ids)}")

# ---------------------------------------------------------------
# 2. Inpatient claims -> cancer type + cost
# ---------------------------------------------------------------
inp = pd.read_csv(f"{RAW_DATA_DIR}/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv")
inp_cancer = inp[inp["DESYNPUF_ID"].isin(cancer_ids)].copy()
inp_cancer["cancer_type"] = inp_cancer.apply(classify_cancer_type, axis=1)
inp_typed = inp_cancer.dropna(subset=["cancer_type"])
inp_cost = inp_typed.groupby("DESYNPUF_ID")["CLM_PMT_AMT"].sum().rename("inpatient_cost")
inp_type_by_bene = (
    inp_typed.groupby(["DESYNPUF_ID", "cancer_type"])["CLM_PMT_AMT"].sum()
    .reset_index().rename(columns={"CLM_PMT_AMT": "cost_in_type"})
)
print(f"Inpatient claims with identified cancer type: {len(inp_typed)}")
print(f"Unique beneficiaries (inpatient): {inp_typed['DESYNPUF_ID'].nunique()}")

# ---------------------------------------------------------------
# 3. Outpatient claims -> cancer type + cost
#    (chunked read given file size: ~790K rows)
# ---------------------------------------------------------------
outp_chunks = []
usecols = ["DESYNPUF_ID", "CLM_PMT_AMT"] + DGNS_COLS
dtype_map = {c: str for c in DGNS_COLS}
dtype_map["DESYNPUF_ID"] = str
dtype_map["CLM_PMT_AMT"] = float
for chunk in pd.read_csv(f"{RAW_DATA_DIR}/DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv",
                          usecols=usecols, dtype=dtype_map, chunksize=100_000):
    chunk_cancer = chunk[chunk["DESYNPUF_ID"].isin(cancer_ids)].copy()
    if len(chunk_cancer):
        chunk_cancer["cancer_type"] = chunk_cancer.apply(classify_cancer_type, axis=1)
        outp_chunks.append(chunk_cancer.dropna(subset=["cancer_type"]))

outp_typed = pd.concat(outp_chunks, ignore_index=True) if outp_chunks else pd.DataFrame()
print(f"Outpatient claims with identified cancer type: {len(outp_typed)}")
print(f"Unique beneficiaries (outpatient): {outp_typed['DESYNPUF_ID'].nunique()}")

outp_type_by_bene = (
    outp_typed.groupby(["DESYNPUF_ID", "cancer_type"])["CLM_PMT_AMT"].sum()
    .reset_index().rename(columns={"CLM_PMT_AMT": "cost_in_type"})
)

# ---------------------------------------------------------------
# 4. Combine inpatient + outpatient: total episode cost per
#    beneficiary, cancer type determined by whichever type has
#    the higher COMBINED cost across both claim sources
# ---------------------------------------------------------------
combined_type = pd.concat([inp_type_by_bene, outp_type_by_bene], ignore_index=True)
combined_type_agg = (
    combined_type.groupby(["DESYNPUF_ID", "cancer_type"])["cost_in_type"].sum()
    .reset_index()
    .sort_values("cost_in_type", ascending=False)
    .drop_duplicates(subset="DESYNPUF_ID", keep="first")
    .rename(columns={"cost_in_type": "type_determining_cost"})
)

# Total episode cost = ALL inpatient + ALL outpatient cost for that
# beneficiary (not just the cost tied to the type-determining claims),
# consistent with an episode-cost definition
inp_total = inp[inp["DESYNPUF_ID"].isin(cancer_ids)].groupby("DESYNPUF_ID")["CLM_PMT_AMT"].sum().rename("total_inpatient")
outp_all_chunks = []
for chunk in pd.read_csv(f"{RAW_DATA_DIR}/DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv",
                          usecols=["DESYNPUF_ID", "CLM_PMT_AMT"],
                          dtype={"DESYNPUF_ID": str, "CLM_PMT_AMT": float},
                          chunksize=200_000):
    c = chunk[chunk["DESYNPUF_ID"].isin(cancer_ids)]
    if len(c):
        outp_all_chunks.append(c)
outp_all = pd.concat(outp_all_chunks, ignore_index=True) if outp_all_chunks else pd.DataFrame(columns=["DESYNPUF_ID","CLM_PMT_AMT"])
outp_total = outp_all.groupby("DESYNPUF_ID")["CLM_PMT_AMT"].sum().rename("total_outpatient")

episode = combined_type_agg[["DESYNPUF_ID", "cancer_type"]].merge(
    inp_total, on="DESYNPUF_ID", how="left").merge(
    outp_total, on="DESYNPUF_ID", how="left")
episode["total_inpatient"] = episode["total_inpatient"].fillna(0)
episode["total_outpatient"] = episode["total_outpatient"].fillna(0)
episode["episode_cost"] = episode["total_inpatient"] + episode["total_outpatient"]

# ---------------------------------------------------------------
# 5. Merge demographics/comorbidity, run regression
# ---------------------------------------------------------------
bene_small = bene[["DESYNPUF_ID", "BENE_BIRTH_DT", "comorbidity_count"]]
df = episode.merge(bene_small, on="DESYNPUF_ID", how="left")
df["approx_age_2009"] = 2009 - df["BENE_BIRTH_DT"].astype(str).str[:4].astype(int)

df.to_csv(f"{OUTPUT_DIR}/multisetting_results.csv", index=False)

print("\n" + "=" * 70)
print("MULTI-SETTING EPISODE COHORT: N = %d" % len(df))
print("=" * 70)
print(df.groupby("cancer_type")["episode_cost"].agg(["count", "mean", "median", "std"]).round(0))
print()
print("Overall mean episode cost (inpatient+outpatient): $%.0f" % df["episode_cost"].mean())
print("Overall median: $%.0f" % df["episode_cost"].median())
print("Mean comorbidity count: %.2f" % df["comorbidity_count"].mean())

df_reg = df[df["episode_cost"] > 0].copy()
df_reg["log_cost"] = np.log(df_reg["episode_cost"])
df_reg["cancer_type"] = pd.Categorical(df_reg["cancer_type"], categories=["Breast", "Lung", "Colorectal"])

model = smf.ols(
    "log_cost ~ C(cancer_type, Treatment(reference='Breast')) "
    "+ comorbidity_count + approx_age_2009",
    data=df_reg,
).fit()

print("\n" + "=" * 70)
print("REGRESSION: MULTI-SETTING EPISODE COST (N=%d)" % len(df_reg))
print("=" * 70)
print(model.summary())

print()
print("=" * 70)
print("% COST IMPACT")
print("=" * 70)
key_vars = {
    "C(cancer_type, Treatment(reference='Breast'))[T.Lung]": "Lung vs. Breast",
    "C(cancer_type, Treatment(reference='Breast'))[T.Colorectal]": "Colorectal vs. Breast",
    "comorbidity_count": "Per comorbidity flag",
    "approx_age_2009": "Per year of age",
}
for var, label in key_vars.items():
    beta = model.params[var]
    pct = (np.exp(beta) - 1) * 100
    pval = model.pvalues[var]
    print(f"{label:30s}: {pct:+7.2f}%   (p={pval:.4f})")
print(f"\nR-squared: {model.rsquared:.4f}")
