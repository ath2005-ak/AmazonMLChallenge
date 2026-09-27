import os
import sys
import time
import numpy as np
import pandas as pd
import polars as pl
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

sys.path.append(r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\code\business_entity_resolution")
from src.preprocessing import clean_business_name, clean_address
from src.metrics import evaluate_candidate_recall

sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\dataset"

print("--- Fast Vectorized TF-IDF Char N-Gram Candidate Blocking ---")
t0 = time.time()

# Load GT
gt_dict = {}
with open(os.path.join(base_dir, "train", "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        s1_id = parts[0]
        m_set = set(parts[1].split(',')) if (len(parts) > 1 and parts[1]) else set()
        gt_dict[s1_id] = m_set

# Sample 30,000 S1 entities
val_s1_ids = set(list(gt_dict.keys())[:30000])
val_gt_dict = {k: gt_dict[k] for k in val_s1_ids}

val_target_ids = set()
for m_set in val_gt_dict.values():
    val_target_ids.update(m_set)

# Load S1 records
s1_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source1.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(val_s1_ids))

# Load S2 and S3 records (first 400k each + GT targets)
s2_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), separator="\t", has_header=True, ignore_errors=True, n_rows=400000)
s3_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), separator="\t", has_header=True, ignore_errors=True, n_rows=400000)

cand_pl = pl.concat([s2_pl, s3_pl])
missing_targets = val_target_ids - set(cand_pl["entity_id"].to_list())
if missing_targets:
    s2_all = pl.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(missing_targets))
    s3_all = pl.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(missing_targets))
    cand_pl = pl.concat([cand_pl, s2_all, s3_all])

s1_df = s1_pl.to_pandas()
cand_df = cand_pl.to_pandas()

s1_df['text'] = (s1_df['business_name'].apply(clean_business_name) + " " + s1_df['business_address'].apply(clean_address)).fillna("")
cand_df['text'] = (cand_df['business_name'].apply(clean_business_name) + " " + cand_df['business_address'].apply(clean_address)).fillna("")

print(f"Data loaded: {len(s1_df):,} S1 and {len(cand_df):,} candidates in {time.time()-t0:.2f}s.")

cand_results = defaultdict(dict)
TOP_K = 25
MIN_SIM = 0.18

for country in s1_df['country'].unique():
    print(f"\nProcessing country: {country}...")
    s1_sub = s1_df[s1_df['country'] == country].reset_index(drop=True)
    cand_sub = cand_df[cand_df['country'] == country].reset_index(drop=True)
    
    print(f"  S1 count: {len(s1_sub):,}, Candidate count: {len(cand_sub):,}")
    
    t_v = time.time()
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2)
    vectorizer.fit(s1_sub['text'].tolist() + cand_sub['text'].tolist())
    
    X_s1 = vectorizer.transform(s1_sub['text'].tolist())
    X_cand = vectorizer.transform(cand_sub['text'].tolist())
    print(f"  TF-IDF matrices constructed in {time.time()-t_v:.2f}s.")
    
    # Fast scipy matrix multiplication
    t_dot = time.time()
    batch_size = 5000
    for b_start in range(0, X_s1.shape[0], batch_size):
        X_batch = X_s1[b_start : b_start + batch_size]
        sim_csr = X_batch.dot(X_cand.T) # CSR matrix [batch_size, num_cand]
        
        # Vectorized extraction per row in CSR matrix
        for r_idx in range(sim_csr.shape[0]):
            s1_id = s1_sub.loc[b_start + r_idx, 'entity_id']
            row_start = sim_csr.indptr[r_idx]
            row_end = sim_csr.indptr[r_idx + 1]
            
            if row_end > row_start:
                cols = sim_csr.indices[row_start:row_end]
                vals = sim_csr.data[row_start:row_end]
                
                if len(vals) > TOP_K:
                    top_indices = np.argpartition(vals, -TOP_K)[-TOP_K:]
                    cols = cols[top_indices]
                    vals = vals[top_indices]
                    
                for c_idx, val in zip(cols, vals):
                    if val >= MIN_SIM:
                        cand_results[s1_id][cand_sub.loc[c_idx, 'entity_id']] = float(val)
                        
    print(f"  Scipy sparse matrix multiplication & Top-K done in {time.time()-t_dot:.2f}s.")

cand_set_dict = {s1_id: set(cands.keys()) for s1_id, cands in cand_results.items()}
res = evaluate_candidate_recall(val_gt_dict, cand_set_dict)

print("\n=== FAST VECTORIZED TF-IDF BLOCKING RESULTS ===")
print(f"Recall Ceiling: {res['recall_ceiling']*100:.2f}%")
print(f"Found Matches: {res['found_matches']:,} / {res['total_matches']:,}")
print(f"Avg Candidates per S1 Entity: {res['avg_candidates_per_s1']:.2f}")

