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
from src.preprocessing import clean_business_name, extract_name_fingerprint, clean_address
from src.metrics import evaluate_candidate_recall

sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Atharva\OneDrive\Documents\Amazon Ml challenge\student_resource\dataset"

print("--- Testing TF-IDF Char N-Gram + Inverted Index Blocking ---")
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

# Sample 20,000 S1 entities for fast iteration
val_s1_ids = set(list(gt_dict.keys())[:20000])
val_gt_dict = {k: gt_dict[k] for k in val_s1_ids}

val_target_ids = set()
for m_set in val_gt_dict.values():
    val_target_ids.update(m_set)

# Load S1 records
s1_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source1.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(val_s1_ids))

# Load S2 and S3 records (first 300k each + GT targets)
s2_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), separator="\t", has_header=True, ignore_errors=True, n_rows=300000)
s3_pl = pl.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), separator="\t", has_header=True, ignore_errors=True, n_rows=300000)

cand_pl = pl.concat([s2_pl, s3_pl])
missing_targets = val_target_ids - set(cand_pl["entity_id"].to_list())
if missing_targets:
    s2_all = pl.read_csv(os.path.join(base_dir, "train", "train_source2.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(missing_targets))
    s3_all = pl.read_csv(os.path.join(base_dir, "train", "train_source3.tsv"), separator="\t", has_header=True, ignore_errors=True).filter(pl.col("entity_id").is_in(missing_targets))
    cand_pl = pl.concat([cand_pl, s2_all, s3_all])

s1_df = s1_pl.to_pandas()
cand_df = cand_pl.to_pandas()

s1_df['clean_name'] = [clean_business_name(n) for n in s1_df['business_name']]
s1_df['clean_addr'] = [clean_address(a) for a in s1_df['business_address']]

cand_df['clean_name'] = [clean_business_name(n) for n in cand_df['business_name']]
cand_df['clean_addr'] = [clean_address(a) for a in cand_df['business_address']]

print(f"Loaded {len(s1_df):,} S1 and {len(cand_df):,} candidates in {time.time()-t0:.2f}s.")

# Multi-pass Blocking per Country
cand_results = defaultdict(set)

for country in s1_df['country'].unique():
    print(f"\nProcessing country: {country}...")
    s1_sub = s1_df[s1_df['country'] == country].reset_index(drop=True)
    cand_sub = cand_df[cand_df['country'] == country].reset_index(drop=True)
    
    print(f"  S1 count: {len(s1_sub):,}, Candidate count: {len(cand_sub):,}")
    
    # 1. Char N-Gram TF-IDF Sparse Matrix Retrieval
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2)
    
    # Combined name + address text for rich vectorization
    s1_texts = (s1_sub['clean_name'] + " " + s1_sub['clean_addr']).tolist()
    cand_texts = (cand_sub['clean_name'] + " " + cand_sub['clean_addr']).tolist()
    
    t_v = time.time()
    vectorizer.fit(s1_texts + cand_texts)
    X_s1 = vectorizer.transform(s1_texts)
    X_cand = vectorizer.transform(cand_texts)
    print(f"  TF-IDF vectorizer fit & transformed in {time.time()-t_v:.2f}s.")
    
    # Query in batches to keep memory low
    batch_size = 5000
    t_m = time.time()
    for i in range(0, X_s1.shape[0], batch_size):
        X_s1_batch = X_s1[i:i+batch_size]
        sim_matrix = X_s1_batch.dot(X_cand.T) # Cosine similarity matrix
        
        # Extract top 20 candidates per S1 row with threshold >= 0.20
        for r_idx in range(sim_matrix.shape[0]):
            s1_id = s1_sub.loc[i + r_idx, 'entity_id']
            row = sim_matrix[r_idx]
            if row.nnz > 0:
                # get top k column indices
                col_indices = row.indices
                data = row.data
                if len(data) > 20:
                    top_k_arg = np.argpartition(data, -20)[-20:]
                    col_indices = col_indices[top_k_arg]
                    data = data[top_k_arg]
                
                for col_idx, sim_val in zip(col_indices, data):
                    if sim_val >= 0.20:
                        cand_results[s1_id].add(cand_sub.loc[col_idx, 'entity_id'])
                        
    print(f"  Sparse matrix retrieval done in {time.time()-t_m:.2f}s.")

res = evaluate_candidate_recall(val_gt_dict, cand_results)
print("\n=== TF-IDF CHAR N-GRAM BLOCKING RESULTS ===")
print(f"Recall Ceiling: {res['recall_ceiling']*100:.2f}%")
print(f"Found Matches: {res['found_matches']:,} / {res['total_matches']:,}")
print(f"Avg Candidates per S1 Entity: {res['avg_candidates_per_s1']:.2f}")

