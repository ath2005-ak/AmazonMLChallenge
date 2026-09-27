import os
import sys
import time
import torch
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

print("--- GPU-Accelerated TF-IDF Char N-Gram Candidate Blocking ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Sample 30,000 S1 entities for benchmarking
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

cand_results = defaultdict(set)
TOP_K = 20
MIN_SIM = 0.18

def scipy_to_torch_sparse(matrix: csr_matrix, device):
    coo = matrix.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape), device=device)

for country in s1_df['country'].unique():
    print(f"\nProcessing country: {country}...")
    s1_sub = s1_df[s1_df['country'] == country].reset_index(drop=True)
    cand_sub = cand_df[cand_df['country'] == country].reset_index(drop=True)
    
    print(f"  S1 count: {len(s1_sub):,}, Candidate count: {len(cand_sub):,}")
    
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2)
    vectorizer.fit(s1_sub['text'].tolist() + cand_sub['text'].tolist())
    
    X_s1 = vectorizer.transform(s1_sub['text'].tolist())
    X_cand = vectorizer.transform(cand_sub['text'].tolist())
    
    print(f"  TF-IDF matrix built. Vocabulary size: {len(vectorizer.vocabulary_):,}")
    
    # Batch matrix multiplication on PyTorch (using CPU/GPU sparse/dense batches)
    batch_size = 2000
    t_batch = time.time()
    
    # Convert cand matrix to torch dense/sparse matrix in GPU or CPU RAM
    X_cand_torch = torch.from_numpy(X_cand.toarray()).float().to(device)
    
    for i in range(0, X_s1.shape[0], batch_size):
        s1_batch_np = X_s1[i:i+batch_size].toarray()
        s1_batch_torch = torch.from_numpy(s1_batch_np).float().to(device)
        
        # Matrix multiplication on GPU: [batch_size, num_cands]
        sim_scores = torch.mm(s1_batch_torch, X_cand_torch.T)
        
        # Get Top-K candidates on GPU
        vals, indices = torch.topk(sim_scores, k=TOP_K, dim=1)
        
        vals_cpu = vals.cpu().numpy()
        indices_cpu = indices.cpu().numpy()
        
        for r_idx in range(len(s1_batch_np)):
            s1_id = s1_sub.loc[i + r_idx, 'entity_id']
            for k_idx in range(TOP_K):
                score = vals_cpu[r_idx, k_idx]
                col_idx = indices_cpu[r_idx, k_idx]
                if score >= MIN_SIM:
                    cand_results[s1_id].add(cand_sub.loc[col_idx, 'entity_id'])
                    
    print(f"  Batched GPU matrix multiplication complete in {time.time()-t_batch:.2f}s.")

res = evaluate_candidate_recall(val_gt_dict, cand_results)
print("\n=== GPU TF-IDF CHAR N-GRAM BLOCKING RESULTS ===")
print(f"Recall Ceiling: {res['recall_ceiling']*100:.2f}%")
print(f"Found Matches: {res['found_matches']:,} / {res['total_matches']:,}")
print(f"Avg Candidates per S1 Entity: {res['avg_candidates_per_s1']:.2f}")

