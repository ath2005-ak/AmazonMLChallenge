import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Set, Tuple
from src.metrics import evaluate_predictions

FEATURE_COLUMNS = [
    "name_exact",
    "fp_exact",
    "name_lev_ratio",
    "name_partial_ratio",
    "name_token_sort",
    "name_token_set",
    "name_jaccard",
    "name_dice",
    "name_char3_jaccard",
    "name_len_diff",
    "name_len_ratio",
    "addr_empty",
    "addr_exact",
    "addr_lev",
    "addr_token_sort",
    "addr_token_set",
    "addr_jaccard",
    "pin_match",
    "pin_mismatch",
    "st_match",
    "is_s2",
    "is_s3",
    "blocking_score",
    "blocking_rank"
]

class EntityMatchingModel:
    def __init__(self, random_state: int = 42):
        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=8,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
        self.best_threshold = 0.50
        self.feature_columns = FEATURE_COLUMNS

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Train LightGBM binary pairwise matching classifier."""
        X_feats = X[self.feature_columns]
        self.model.fit(X_feats, y)
        print("Model training complete.")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probability of true match for candidate pairs."""
        X_feats = X[self.feature_columns]
        return self.model.predict_proba(X_feats)[:, 1]

    def calibrate_threshold(
        self,
        df_val_pairs: pd.DataFrame,
        probs: np.ndarray,
        val_gt_dict: Dict[str, Set[str]],
        threshold_range: np.ndarray = np.arange(0.15, 0.90, 0.02)
    ) -> Tuple[float, dict]:
        """
        Line search over probability thresholds to find T* that maximizes macro F_0.5 score.
        """
        best_score = -1.0
        best_t = 0.50
        best_metrics = {}
        
        df_eval = df_val_pairs[['s1_id', 'cand_id']].copy()
        df_eval['prob'] = probs
        
        # Group candidates per S1 entity for fast evaluation
        grouped = df_eval.groupby('s1_id')
        
        print("\n=== CALIBRATING OPTIMAL F_0.5 THRESHOLD ===")
        for t in threshold_range:
            t_round = float(np.round(t, 2))
            
            # Predict candidates with probability >= t
            filtered = df_eval[df_eval['prob'] >= t_round]
            pred_dict = {s1_id: set(group['cand_id']) for s1_id, group in filtered.groupby('s1_id')}
            
            # Include all validation S1 entities (singletons get empty set)
            for s1_id in val_gt_dict:
                if s1_id not in pred_dict:
                    pred_dict[s1_id] = set()
                    
            m = evaluate_predictions(val_gt_dict, pred_dict)
            score = m['macro_f05']
            
            if score > best_score:
                best_score = score
                best_t = t_round
                best_metrics = m
                
            print(f"  Threshold {t_round:.2f} -> Macro F_0.5: {score:.4f} (Prec: {m['mean_precision']:.4f}, Rec: {m['mean_recall']:.4f}, Singleton Acc: {m['singleton_accuracy']*100:.2f}%)")
            
        self.best_threshold = best_t
        print(f"\nOptimal Threshold: {best_t:.2f} with Validation Macro F_0.5 = {best_score:.4f}")
        return best_t, best_metrics
