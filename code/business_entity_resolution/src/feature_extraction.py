import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

def get_char_ngrams(s: str, n: int = 3) -> set:
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    union_len = len(set1.union(set2))
    return len(set1.intersection(set2)) / union_len if union_len > 0 else 0.0

def dice_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    denom = len(set1) + len(set2)
    return (2.0 * len(set1.intersection(set2))) / denom if denom > 0 else 0.0

def extract_pairwise_features(
    s1_norm_name: str,
    s1_fp: str,
    s1_norm_addr: str,
    s1_pin: str,
    s1_st: str,
    cand_id: str,
    cand_norm_name: str,
    cand_fp: str,
    cand_norm_addr: str,
    cand_pin: str,
    cand_st: str,
    blocking_score: float = 0.0,
    blocking_rank: int = 1
) -> dict:
    """
    Extracts high-dimensional pairwise similarity features between S1 entity and candidate record.
    """
    # Name Features
    n1, n2 = s1_norm_name, cand_norm_name
    name_exact = 1.0 if (n1 and n1 == n2) else 0.0
    fp_exact = 1.0 if (s1_fp and s1_fp == cand_fp) else 0.0
    
    lev_ratio = fuzz.ratio(n1, n2) / 100.0 if (n1 and n2) else 0.0
    partial_ratio = fuzz.partial_ratio(n1, n2) / 100.0 if (n1 and n2) else 0.0
    token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0 if (n1 and n2) else 0.0
    token_set = fuzz.token_set_ratio(n1, n2) / 100.0 if (n1 and n2) else 0.0
    
    t1_set = set(n1.split())
    t2_set = set(n2.split())
    name_jaccard = jaccard_similarity(t1_set, t2_set)
    name_dice = dice_similarity(t1_set, t2_set)
    
    g1_set = get_char_ngrams(n1, 3)
    g2_set = get_char_ngrams(n2, 3)
    char3_jaccard = jaccard_similarity(g1_set, g2_set)
    
    len1, len2 = len(n1), len(n2)
    len_diff = float(abs(len1 - len2))
    len_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 0.0
    
    # Address Features
    a1, a2 = s1_norm_addr, cand_norm_addr
    addr_empty = 1.0 if (not a1 or not a2) else 0.0
    addr_exact = 1.0 if (a1 and a1 == a2) else 0.0
    
    addr_lev = fuzz.ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
    addr_token_sort = fuzz.token_sort_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
    addr_token_set = fuzz.token_set_ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0
    
    at1_set = set(a1.split())
    at2_set = set(a2.split())
    addr_jaccard = jaccard_similarity(at1_set, at2_set)
    
    # Pincode / Street Num Features
    pin_match = 1.0 if (s1_pin and cand_pin and s1_pin == cand_pin) else 0.0
    pin_mismatch = 1.0 if (s1_pin and cand_pin and s1_pin != cand_pin) else 0.0
    
    st_match = 1.0 if (s1_st and cand_st and s1_st == cand_st) else 0.0
    
    # Metadata Features
    is_s2 = 1.0 if cand_id.startswith('S2-') else 0.0
    is_s3 = 1.0 if cand_id.startswith('S3-') else 0.0
    
    return {
        "name_exact": name_exact,
        "fp_exact": fp_exact,
        "name_lev_ratio": lev_ratio,
        "name_partial_ratio": partial_ratio,
        "name_token_sort": token_sort,
        "name_token_set": token_set,
        "name_jaccard": name_jaccard,
        "name_dice": name_dice,
        "name_char3_jaccard": char3_jaccard,
        "name_len_diff": len_diff,
        "name_len_ratio": len_ratio,
        "addr_empty": addr_empty,
        "addr_exact": addr_exact,
        "addr_lev": addr_lev,
        "addr_token_sort": addr_token_sort,
        "addr_token_set": addr_token_set,
        "addr_jaccard": addr_jaccard,
        "pin_match": pin_match,
        "pin_mismatch": pin_mismatch,
        "st_match": st_match,
        "is_s2": is_s2,
        "is_s3": is_s3,
        "blocking_score": float(blocking_score),
        "blocking_rank": float(blocking_rank)
    }
