import os
import sys
import time
import numpy as np
import pandas as pd
import polars as pl

sys.path.append(r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\code\business_entity_resolution")
from src.preprocessing import clean_business_name, clean_address, extract_pincode
from src.blocking import CandidateBlocker, normalize_string, extract_name_fp, extract_street_num, extract_addr_tokens
from src.feature_extraction import extract_pairwise_features
from src.metrics import evaluate_candidate_recall, evaluate_predictions
from src.model import EntityMatchingModel

sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\dataset"

print("==========================================================")
print("===   ENHANCED TRAINING & VALIDATION EXPERIMENT        ===")
print("==========================================================")
t0 = time.time()

# 1. Load Ground Truth
print("1. Loading Ground Truth...")
gt_dict = {}
with open(os.path.join(base_dir, "train", "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        s1_id = parts[0]
        m_set = set(parts[1].split(',')) if (len(parts) > 1 and parts[1]) else set()
        gt_dict[s1_id] = m_set

all_s1_ids = list(gt_dict.keys())
np.random.seed(42)
np.random.shuffle(all_s1_ids)

# 100k S1 sample split into 80k train / 20k validation
sample_s1_ids = all_s1_ids[:100000]
train_s1_ids = set(sample_s1_ids[:80000])
val_s1_ids = set(sample_s1_ids[80000:])

train_gt_dict = {k: gt_dict[k] for k in train_s1_ids}
val_gt_dict = {k: gt_dict[k] for k in val_s1_ids}

print(f"   Total sampled S1 entities: {len(sample_s1_ids):,}")
print(f"   Train S1 count: {len(train_s1_ids):,} (Singletons: {sum(1 for v in train_gt_dict.values() if len(v)==0):,})")
print(f"   Val S1 count:   {len(val_s1_ids):,} (Singletons: {sum(1 for v in val_gt_dict.values() if len(v)==0):,})")

# 2. Load Source Datasets via Polars
print("\n2. Loading Source Files...")
s1_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source1.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(set(sample_s1_ids)))

# Determine target S2/S3 IDs needed for candidate pool
target_cand_ids = set()
for m_set in train_gt_dict.values():
    target_cand_ids.update(m_set)
for m_set in val_gt_dict.values():
    target_cand_ids.update(m_set)

# Load S2 and S3 (first 600k each + target GT matches)
s2_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), separator="\t", has_header=True, ignore_errors=True, n_rows=600000)
s3_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), separator="\t", has_header=True, ignore_errors=True, n_rows=600000)

cand_pl = pl.concat([s2_pl, s3_pl])
missing_cand_ids = target_cand_ids - set(cand_pl["entity_id"].to_list())

if missing_cand_ids:
    print(f"   Loading {len(missing_cand_ids):,} missing GT target records...")
    s2_miss = pl.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(missing_cand_ids))
    s3_miss = pl.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(missing_cand_ids))
    cand_pl = pl.concat([cand_pl, s2_miss, s3_miss])

s1_df = s1_pl.to_pandas()
cand_df = cand_pl.to_pandas()

print(f"   Loaded S1 records: {len(s1_df):,}")
print(f"   Loaded Candidate pool: {len(cand_df):,}")

# 3. Preprocessing Data
print("\n3. Preprocessing text fields...")
t_prep = time.time()

s1_df['norm_name'] = [normalize_string(n) for n in s1_df['business_name']]
s1_df['fp'] = [extract_name_fp(n) for n in s1_df['business_name']]
s1_df['norm_addr'] = [clean_address(a) for a in s1_df['business_address']]
s1_df['pin'] = [extract_pincode(a) for a in s1_df['business_address']]
s1_df['st_num'] = [extract_street_num(a) for a in s1_df['business_address']]
s1_df['addr_toks'] = [extract_addr_tokens(a) for a in s1_df['business_address']]

cand_df['norm_name'] = [normalize_string(n) for n in cand_df['business_name']]
cand_df['fp'] = [extract_name_fp(n) for n in cand_df['business_name']]
cand_df['norm_addr'] = [clean_address(a) for a in cand_df['business_address']]
cand_df['pin'] = [extract_pincode(a) for a in cand_df['business_address']]
cand_df['st_num'] = [extract_street_num(a) for a in cand_df['business_address']]
cand_df['addr_toks'] = [extract_addr_tokens(a) for a in cand_df['business_address']]

print(f"   Preprocessing complete in {time.time()-t_prep:.2f}s.")

# Map candidate metadata by entity_id
cand_map = {}
for row in cand_df.itertuples(index=False):
    cand_map[row.entity_id] = row

# 4. Multi-Pass Candidate Generation
print("\n4. Multi-Pass Blocking & Candidate Generation...")
t_block = time.time()

blocker = CandidateBlocker(top_k=30)
cand_results = blocker.generate_candidates(s1_df, cand_df)

print(f"   Candidate generation complete in {time.time()-t_block:.2f}s.")

# Candidate Recall Evaluation
cand_set_dict = {s1_id: set(cands.keys()) for s1_id, cands in cand_results.items()}

train_cand_eval = evaluate_candidate_recall(train_gt_dict, cand_set_dict)
val_cand_eval = evaluate_candidate_recall(val_gt_dict, cand_set_dict)

print(f"\n   [BLOCKING RECALL] Train Recall Ceiling: {train_cand_eval['recall_ceiling']*100:.2f}% (Avg Cand: {train_cand_eval['avg_candidates_per_s1']:.2f})")
print(f"   [BLOCKING RECALL] Val Recall Ceiling:   {val_cand_eval['recall_ceiling']*100:.2f}% (Avg Cand: {val_cand_eval['avg_candidates_per_s1']:.2f})")

# 5. Feature Extraction for Train & Val Pairs
print("\n5. Extracting Pairwise Features for Classifier...")
t_feat = time.time()

def build_feature_dataset(s1_subset_df, target_gt_dict):
    rows = []
    labels = []
    
    for s1_row in s1_subset_df.itertuples(index=False):
        s1_id = s1_row.entity_id
        cands_dict = cand_results.get(s1_id, {})
        true_gt_set = target_gt_dict.get(s1_id, set())
        
        sorted_cands = sorted(cands_dict.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (cand_id, b_score) in enumerate(sorted_cands, 1):
            cand_row = cand_map.get(cand_id)
            if not cand_row:
                continue
                
            is_match = 1 if cand_id in true_gt_set else 0
            
            feat = extract_pairwise_features(
                s1_norm_name=s1_row.norm_name,
                s1_fp=s1_row.fp,
                s1_norm_addr=s1_row.norm_addr,
                s1_pin=s1_row.pin,
                s1_st=s1_row.st_num,
                cand_id=cand_id,
                cand_norm_name=cand_row.norm_name,
                cand_fp=cand_row.fp,
                cand_norm_addr=cand_row.norm_addr,
                cand_pin=cand_row.pin,
                cand_st=cand_row.st_num,
                blocking_score=b_score,
                blocking_rank=rank
            )
            feat['s1_id'] = s1_id
            feat['cand_id'] = cand_id
            rows.append(feat)
            labels.append(is_match)
            
    df_feats = pd.DataFrame(rows)
    return df_feats, np.array(labels)

train_s1_df = s1_df[s1_df['entity_id'].isin(train_s1_ids)].reset_index(drop=True)
val_s1_df = s1_df[s1_df['entity_id'].isin(val_s1_ids)].reset_index(drop=True)

df_train_feats, y_train = build_feature_dataset(train_s1_df, train_gt_dict)
df_val_feats, y_val = build_feature_dataset(val_s1_df, val_gt_dict)

print(f"   Train candidate pairs: {len(df_train_feats):,} (Positive matches: {sum(y_train):,})")
print(f"   Val candidate pairs:   {len(df_val_feats):,} (Positive matches: {sum(y_val):,})")
print(f"   Feature extraction complete in {time.time()-t_feat:.2f}s.")

# 6. Train LightGBM Matching Classifier
print("\n6. Training LightGBM Pairwise Classifier...")
model = EntityMatchingModel(random_state=42)
model.fit(df_train_feats, y_train)

# Feature Importance
imp = pd.DataFrame({'feature': model.feature_columns, 'importance': model.model.feature_importances_}).sort_values('importance', ascending=False)
print("\n   Top 10 Feature Importances:")
for _, r in imp.head(10).iterrows():
    print(f"     {r['feature']:<20}: {r['importance']}")

# 7. Predict & Calibrate F_0.5 Threshold on Validation Set
print("\n7. Evaluating & Calibrating Optimal F_0.5 Threshold...")
val_probs = model.predict_proba(df_val_feats)

best_t, val_metrics = model.calibrate_threshold(df_val_feats, val_probs, val_gt_dict)

print("\n==========================================================")
print("===              FINAL EXPERIMENT RESULTS              ===")
print("==========================================================")
print(f" Validation Macro F_0.5 Score : {val_metrics['macro_f05']:.4f}")
print(f" Precision                  : {val_metrics['mean_precision']:.4f}")
print(f" Recall                     : {val_metrics['mean_recall']:.4f}")
print(f" Singleton Accuracy          : {val_metrics['singleton_accuracy']*100:.2f}%")
print(f" Optimal Threshold (T*)     : {best_t:.2f}")
print(f" Total Execution Time       : {(time.time()-t0)/60:.2f} minutes")
print("==========================================================")

