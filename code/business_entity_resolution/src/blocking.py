import re
import unicodedata
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

DOMAIN_PATTERN = re.compile(r'\b([\w\-]+)\.(com|org|net|in|co\.in|us|fr)\b', flags=re.IGNORECASE)
LEGAL_SUFFIXES = re.compile(r'\b(pvt|private|ltd|limited|inc|incorporated|corp|corporation|llc|llp|co|company|sarl|sas|gmbh|s\.a\.r\.l|s\.a\.s|s\.a|n\.v|b\.v|aka|f\/k\/a)\b', flags=re.IGNORECASE)

STOP_WORDS = {
    'store', 'shop', 'company', 'ltd', 'pvt', 'inc', 'corp', 'services', 'service',
    'center', 'centre', 'group', 'enterprises', 'traders', 'trading', 'solutions',
    'technologies', 'private', 'limited', 'corporation', 'sales', 'mart', 'market',
    'and', 'the', 'for', 'with', 'from', 'near', 'opp', 'road', 'street', 'main'
}

def normalize_string(s: str) -> str:
    """ASCII Unidecode, domain stripping, lowercasing, and whitespace collapse."""
    if not s or not isinstance(s, str):
        return ""
    s = DOMAIN_PATTERN.sub(r'\1', s)
    s = unicodedata.normalize('NFKD', s)
    s = "".join([c for c in s if not unicodedata.combining(c)]).lower()
    s = s.replace('&', ' and ')
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def extract_name_fp(name: str) -> str:
    norm = normalize_string(name)
    norm_no_legal = LEGAL_SUFFIXES.sub('', norm)
    tokens = sorted([t for t in norm_no_legal.split() if len(t) > 1])
    return " ".join(tokens)

def extract_street_num(addr: str) -> str:
    if not addr or not isinstance(addr, str):
        return ""
    matches = re.findall(r'\b\d{2,6}\b', addr)
    return matches[0] if matches else ""

def extract_addr_tokens(addr: str) -> list:
    norm = normalize_string(addr)
    tokens = [t for t in norm.split() if len(t) >= 4 and not t.isdigit()]
    return tokens[:3]

def get_record_keys(norm_name: str, fp: str, st_num: str, addr_toks: list) -> list:
    keys = []
    if fp and len(fp) >= 3:
        keys.append(('fp', fp))
        
    words = [w for w in norm_name.split() if len(w) >= 3 and w not in STOP_WORDS]
    for w in words:
        keys.append(('w', w))
        
    if st_num and len(st_num) >= 2:
        keys.append(('st', st_num))
        for atok in addr_toks:
            keys.append(('st_at', f"{st_num}_{atok}"))
            
    return keys

class CandidateBlocker:
    def __init__(self, top_k: int = 30):
        self.top_k = top_k
        self.max_limits = {
            'fp': 1500,
            'w': 300,
            'st_at': 200,
            'st': 500
        }
        
    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        cand_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Hybrid Blocking: Inverted Index Keys + Filtered TF-IDF Word/Char Cosine Retrieval.
        Returns: {s1_id: {cand_id: blocking_score}}
        """
        cand_results = defaultdict(dict)
        
        for country in s1_df['country'].unique():
            cand_sub = cand_df[cand_df['country'] == country].reset_index(drop=True)
            s1_sub = s1_df[s1_df['country'] == country].reset_index(drop=True)
            
            # --- Pass A: Multi-Key Inverted Index ---
            idx_dict = defaultdict(list)
            for row in cand_sub.itertuples(index=False):
                keys = get_record_keys(row.norm_name, row.fp, row.st_num, row.addr_toks)
                for k_type, k_val in keys:
                    idx_dict[(k_type, k_val)].append(row.entity_id)
                    
            for s1_row in s1_sub.itertuples(index=False):
                s1_id = s1_row.entity_id
                keys = get_record_keys(s1_row.norm_name, s1_row.fp, s1_row.st_num, s1_row.addr_toks)
                
                scores = Counter()
                for k_type, k_val in keys:
                    bucket = idx_dict.get((k_type, k_val), [])
                    max_lim = self.max_limits.get(k_type, 150)
                    if 0 < len(bucket) <= max_lim:
                        w_score = 5.0 if k_type == 'fp' else (3.0 if k_type == 'st_at' else (2.0 if k_type == 'w' else 0.8))
                        for cid in bucket:
                            scores[cid] += w_score
                            
                if scores:
                    for cid, sc in scores.most_common(self.top_k):
                        cand_results[s1_id][cid] = float(sc)
                        
            # --- Pass B: TF-IDF Cosine Retrieval for zero-hit S1 records ---
            # Only apply to S1 records that got 0 candidates from inverted index,
            # and skip entirely if the candidate slice is too large (avoids OOM).
            TFIDF_CAND_LIMIT = 50000
            if len(cand_sub) <= TFIDF_CAND_LIMIT:
                # Find S1 rows with 0 inverted-index candidates
                zero_hit_mask = [
                    s1_sub.loc[i, 'entity_id'] not in cand_results
                    for i in range(len(s1_sub))
                ]
                s1_zero_indices = [i for i, z in enumerate(zero_hit_mask) if z]
                
                if s1_zero_indices:
                    all_texts = (s1_sub['norm_name'] + " " + s1_sub['norm_addr']).tolist()
                    cand_texts = (cand_sub['norm_name'] + " " + cand_sub['norm_addr']).tolist()
                    
                    vectorizer = TfidfVectorizer(
                        analyzer='char_wb', ngram_range=(4, 5),
                        min_df=3, max_df=0.20
                    )
                    vectorizer.fit(all_texts + cand_texts)
                    
                    # Only transform zero-hit S1 rows
                    zero_texts = [all_texts[i] for i in s1_zero_indices]
                    X_s1_zero = vectorizer.transform(zero_texts)
                    X_cand_mat = vectorizer.transform(cand_texts)
                    cand_ids_list = cand_sub['entity_id'].tolist()
                    
                    # Process row-by-row to avoid large dense intermediates
                    for local_idx, global_idx in enumerate(s1_zero_indices):
                        s1_id = s1_sub.loc[global_idx, 'entity_id']
                        row_vec = X_s1_zero[local_idx]
                        sim_row = row_vec.dot(X_cand_mat.T)
                        
                        if sim_row.nnz == 0:
                            continue
                        
                        cols = sim_row.indices
                        vals = sim_row.data
                        
                        if len(vals) > 15:
                            top_indices = np.argpartition(vals, -15)[-15:]
                            cols = cols[top_indices]
                            vals = vals[top_indices]
                        
                        for c_idx, val in zip(cols, vals):
                            if val >= 0.25:
                                cid = cand_ids_list[c_idx]
                                curr_sc = cand_results[s1_id].get(cid, 0.0)
                                cand_results[s1_id][cid] = max(curr_sc, float(val * 10.0))
                                
        return cand_results
