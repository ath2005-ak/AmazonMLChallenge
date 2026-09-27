import numpy as np

def compute_f05_single(gt_set: set, pred_set: set) -> float:
    """
    Computes F_0.5 score for a single S1 entity.
    
    Singletons (gt_set is empty):
      - 1.0 if pred_set is empty
      - 0.0 if pred_set is non-empty
    
    Non-singletons (gt_set is non-empty):
      - 0.0 if pred_set is empty
      - (1.25 * Precision * Recall) / (0.25 * Precision + Recall) otherwise
    """
    if len(gt_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    
    if len(pred_set) == 0:
        return 0.0
    
    tp = len(gt_set.intersection(pred_set))
    if tp == 0:
        return 0.0
    
    precision = tp / len(pred_set)
    recall = tp / len(gt_set)
    
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    
    return (1.25 * precision * recall) / denom

def evaluate_predictions(gt_dict: dict, pred_dict: dict) -> dict:
    """
    Computes macro-averaged F_0.5, Precision, Recall, and Singleton statistics.
    
    Args:
        gt_dict: {s1_id: set(matched_ids)}
        pred_dict: {s1_id: set(predicted_matched_ids)}
        
    Returns:
        dict with metrics
    """
    f05_scores = []
    precisions = []
    recalls = []
    
    singleton_count = 0
    singleton_correct = 0
    non_singleton_count = 0
    
    for s1_id, gt_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        
        score = compute_f05_single(gt_set, pred_set)
        f05_scores.append(score)
        
        if len(gt_set) == 0:
            singleton_count += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            non_singleton_count += 1
            if len(pred_set) > 0:
                tp = len(gt_set.intersection(pred_set))
                precisions.append(tp / len(pred_set))
                recalls.append(tp / len(gt_set))
            else:
                precisions.append(0.0)
                recalls.append(0.0)
                
    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    mean_precision = float(np.mean(precisions)) if precisions else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    singleton_acc = float(singleton_correct / singleton_count) if singleton_count > 0 else 1.0
    
    return {
        "macro_f05": macro_f05,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "singleton_count": singleton_count,
        "singleton_accuracy": singleton_acc,
        "total_eval_entities": len(gt_dict)
    }

def evaluate_candidate_recall(gt_dict: dict, cand_dict: dict) -> dict:
    """
    Evaluates blocking quality (Recall Ceiling & Average Reduction Ratio).
    """
    total_true_matches = 0
    found_in_candidates = 0
    cand_counts = []
    
    for s1_id, gt_set in gt_dict.items():
        cand_set = cand_dict.get(s1_id, set())
        cand_counts.append(len(cand_set))
        if len(gt_set) > 0:
            total_true_matches += len(gt_set)
            found_in_candidates += len(gt_set.intersection(cand_set))
            
    recall_ceiling = (found_in_candidates / total_true_matches) if total_true_matches > 0 else 1.0
    avg_candidates = float(np.mean(cand_counts)) if cand_counts else 0.0
    
    return {
        "recall_ceiling": float(recall_ceiling),
        "found_matches": found_in_candidates,
        "total_matches": total_true_matches,
        "avg_candidates_per_s1": avg_candidates
    }
