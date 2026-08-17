# cms-claims-cost-driver-pipeline
Claims-based cancer cost-driver pipeline built on CMS's official synthetic Medicare claims data (DE-SynPUF)- real ICD-9 coding, real claims structure.
# Claims-Based Cancer Cost-Driver Pipeline (CMS DE-SynPUF)

## What this project actually does

Health insurers use patient claims data (hospital visits, doctor visits, diagnoses, costs) to figure out what drives the cost of treating a disease. This project builds that kind of analysis for cancer patients, using real government Medicare claims data.

The question I set out to answer: **when you only look at hospital stays, do you get a different answer than when you also include outpatient visits (doctor visits, labs, imaging, chemo infusions)?**

**Short answer: yes, dramatically.**

Using only inpatient (hospital) data, the model found that a patient's overall health burden (comorbidities like diabetes, heart failure, etc.) had no measurable effect on cost. That seemed wrong — health status should matter.

Adding outpatient data (visits outside the hospital) fixed it: comorbidity burden became the single strongest predictor of cost in the model, and the model explained 3x more of what actually drives cost.

**The takeaway: if you only look at part of a patient's care, you can reach the wrong conclusion about what actually drives their cost.** That's a real, useful lesson for anyone doing healthcare cost analysis.

## The numbers

| | Hospital claims only | Hospital + outpatient claims |
|---|---|---|
| Patients in the analysis | 1,053 | 4,016 |
| How much the model explains (R²) | 9% | 28% |
| Does health burden predict cost? | No (not statistically significant) | **Yes — the strongest factor in the model** |

## The data

I used the [CMS Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF)](https://www.cms.gov/) — a free, publicly available Medicare claims dataset that CMS built specifically so people could practice this kind of analysis. It has the real structure and coding of actual Medicare claims (real diagnosis codes, real claim types), but the patient-level numbers are synthetic, so no real patient's information is in this data.

## What's in this repo

- **`code/`** — the two Python scripts: one using only hospital claims, one using hospital + outpatient claims combined
- **`data/`** — the resulting datasets each script produced
- **`docs/`** — three write-ups: a methods paper for each version, plus a version formatted for journal submission

## How to run it yourself

1. Download DE-SynPUF Sample 1 from cms.gov (free, no application needed) — you need the Beneficiary Summary, Inpatient Claims, and Outpatient Claims files
2. Put those files in `data/raw/`
3. `pip install pandas numpy statsmodels`
4. Run `code/01_inpatient_only_pipeline.py`, then `code/02_multisetting_pipeline.py`

## Tools used

Python — pandas, numpy, statsmodels

## About me

Shruthi, MD, MHA -OSU DHA Student. Background in CMS Medicare Advantage regulatory reporting.

## One important note

This uses CMS's *synthetic* Medicare data — real file structure and coding, but not real patients. CMS itself says this data shouldn't be used to draw real-world conclusions about actual Medicare costs. This project is a demonstration of the analysis method, not a claim about real-world cancer costs.

