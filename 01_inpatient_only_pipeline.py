"""
Real CMS DE-SynPUF pipeline: identify a cancer cohort using actual ICD-9
diagnosis codes, compute comorbidity burden, and build a claims-based
episode-cost analysis -- following the same general methodology as
Sagar, Lin, Castel (2017), now applied to CMS's official synthetic
Medicare claims data (real file structure, real ICD-9 codes) rather
than a fully hand-simulated dataset.

Data source: CMS 2008-2010 Data Entrepreneurs' Synthetic Public Use
File (DE-SynPUF), Sample 1. This is CMS's own official synthetic
Medicare dataset -- real claims file structure and real ICD-9 codes,
with beneficiary-level values synthesized/coarsened by CMS to prevent
re-identification. Per CMS's own documentation, this file has "very
limited inferential research value" and should not be used to draw
real-world conclusions -- it is used here for pipeline-building and
methods demonstration purposes.
"""

import pandas as pd
import numpy as np

pd.set_option("display.width", 160)

# Place the three downloaded DE-SynPUF Sample 1 files in this folder
# before running (see README.md for the CMS download link)
RAW_DATA_DIR = "./data/raw"
OUTPUT_DIR = "./data"

# ---------------------------------------------------------------
# 1. Load beneficiary summary (has CMS's own pre-computed chronic
#    condition flags -- SP_CNCR = cancer flag)
# ---------------------------------------------------------------
bene = pd.read_csv(f"{RAW_DATA_DIR}/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv")

print("=" * 70)
print("BENEFICIARY FILE: %d beneficiaries" % len(bene))
print("=" * 70)

# Chronic condition flags: 1 = yes, 2 = no (per CMS coding)
chronic_cols = ["SP_ALZHDMTA", "SP_CHF", "SP_CHRNKIDN", "SP_CNCR", "SP_COPD",
                 "SP_DEPRESSN", "SP_DIABETES", "SP_ISCHMCHT", "SP_OSTEOPRS",
                 "SP_RA_OA", "SP_STRKETIA"]

for col in chronic_cols:
    pct = (bene[col] == 1).mean() * 100
    print(f"  {col:15s}: {pct:5.1f}% flagged")

cancer_cohort_ids = set(bene.loc[bene["SP_CNCR"] == 1, "DESYNPUF_ID"])
print(f"\nBeneficiaries with CMS cancer flag (SP_CNCR=1): {len(cancer_cohort_ids)}")

# ---------------------------------------------------------------
# 2. Compute a simple comorbidity count per beneficiary (0-11 scale)
#    using CMS's own pre-flagged chronic conditions -- a practical,
#    real-data analog to a Charlson-style burden score, built from
#    fields CMS provides directly rather than requiring an ICD-9
#    crosswalk for this beneficiary-level file.
# ---------------------------------------------------------------
bene["comorbidity_count"] = (bene[chronic_cols] == 1).sum(axis=1)

print("\nComorbidity count distribution (0-11 chronic conditions):")
print(bene["comorbidity_count"].describe().round(2))

# ---------------------------------------------------------------
# 3. Load inpatient claims for the cancer cohort and build a
#    real episode-cost measure: total inpatient claim payment
#    amount per beneficiary across the 2008-2010 window.
# ---------------------------------------------------------------
inp = pd.read_csv(f"{RAW_DATA_DIR}/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv")

print("\n" + "=" * 70)
print("INPATIENT CLAIMS FILE: %d claims" % len(inp))
print("=" * 70)

inp_cancer = inp[inp["DESYNPUF_ID"].isin(cancer_cohort_ids)].copy()
print(f"Inpatient claims belonging to cancer-flagged beneficiaries: {len(inp_cancer)}")

# Real ICD-9 cancer diagnosis code ranges (140-239 = neoplasms in ICD-9-CM)
# Breast: 174-175, Lung: 162, Colorectal: 153-154
def classify_cancer_type(row):
    codes = [str(row[c]) for c in
             [f"ICD9_DGNS_CD_{i}" for i in range(1, 11)] if pd.notna(row.get(c))]
    for code in codes:
        code3 = code[:3]
        if code3 in ("174", "175"):
            return "Breast"
        if code3 == "162":
            return "Lung"
        if code3 in ("153", "154"):
            return "Colorectal"
    return None

inp_cancer["cancer_type"] = inp_cancer.apply(classify_cancer_type, axis=1)
typed = inp_cancer.dropna(subset=["cancer_type"])
print(f"\nInpatient claims with a matching breast/lung/colorectal ICD-9 code: {len(typed)}")
print(typed["cancer_type"].value_counts())

# ---------------------------------------------------------------
# 4. Build per-beneficiary episode cost: sum of inpatient claim
#    payments for beneficiaries with an identified cancer-type claim
# ---------------------------------------------------------------
episode_cost = (
    typed.groupby(["DESYNPUF_ID", "cancer_type"])["CLM_PMT_AMT"]
    .sum()
    .reset_index()
    .rename(columns={"CLM_PMT_AMT": "total_inpatient_cost"})
)

# Some beneficiaries may have claims classified under more than one
# cancer type (multiple admissions, different codes) -- keep the
# type with the higher total cost for a clean one-row-per-patient set
episode_cost = (
    episode_cost.sort_values("total_inpatient_cost", ascending=False)
    .drop_duplicates(subset="DESYNPUF_ID", keep="first")
)

# Merge with beneficiary comorbidity burden and demographics
bene_small = bene[["DESYNPUF_ID", "BENE_BIRTH_DT", "BENE_SEX_IDENT_CD",
                    "comorbidity_count", "SP_CHF", "SP_CHRNKIDN", "SP_DIABETES"]]
analysis_df = episode_cost.merge(bene_small, on="DESYNPUF_ID", how="left")

analysis_df["birth_year"] = analysis_df["BENE_BIRTH_DT"].astype(str).str[:4].astype(int)
analysis_df["approx_age_2009"] = 2009 - analysis_df["birth_year"]

print("\n" + "=" * 70)
print("FINAL ANALYSIS COHORT: %d beneficiaries with identifiable" % len(analysis_df))
print("breast/lung/colorectal cancer inpatient claims")
print("=" * 70)
print(analysis_df.groupby("cancer_type")["total_inpatient_cost"].agg(
    ["count", "mean", "median", "std"]).round(0))

print("\nOverall mean inpatient cost (identified cancer cohort): $%.0f" %
      analysis_df["total_inpatient_cost"].mean())
print("Mean comorbidity count in this cohort: %.2f" %
      analysis_df["comorbidity_count"].mean())
print("Mean approx. age (as of 2009): %.1f" % analysis_df["approx_age_2009"].mean())

analysis_df.to_csv(f"{OUTPUT_DIR}/inpatient_only_results.csv", index=False)

# ---------------------------------------------------------------
# 5. Simple regression using REAL data: does comorbidity burden
#    and cancer type predict inpatient cost in this real (synthetic
#    Medicare-format) cohort?
# ---------------------------------------------------------------
import statsmodels.formula.api as smf

reg_df = analysis_df[analysis_df["total_inpatient_cost"] > 0].copy()
reg_df["log_cost"] = np.log(reg_df["total_inpatient_cost"])
reg_df["cancer_type"] = pd.Categorical(reg_df["cancer_type"],
                                        categories=["Breast", "Lung", "Colorectal"])

if len(reg_df) > 30:  # only run if enough sample size
    model = smf.ols(
        "log_cost ~ C(cancer_type, Treatment(reference='Breast')) "
        "+ comorbidity_count + approx_age_2009",
        data=reg_df,
    ).fit()
    print("\n" + "=" * 70)
    print("REGRESSION ON REAL CMS DE-SYNPUF DATA (N=%d)" % len(reg_df))
    print("=" * 70)
    print(model.summary())
else:
    print(f"\nSample size too small in this single sample (N={len(reg_df)}) "
          f"for a stable regression -- this is expected with only Sample 1 "
          f"and only the inpatient file loaded. Combining multiple DE-SynPUF "
          f"samples and adding outpatient/carrier claims would substantially "
          f"increase cohort size for a production version of this analysis.")
