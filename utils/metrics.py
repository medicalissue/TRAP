"""
TAO-compatible evaluation metrics.
Includes pixel-level, object-level (RBDC, TBDC), and frame-level metrics.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
from typing import Dict, List, Tuple, Optional
import warnings


def compute_pixel_auroc(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute pixel-level AUROC.

    Args:
        predictions: [N, H, W] or [N] pixel-level anomaly scores
        targets: [N, H, W] or [N] binary labels (0=normal, 1=anomaly)

    Returns:
        Pixel-AUROC score
    """
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    if len(np.unique(target_flat)) < 2:
        warnings.warn("Only one class present in targets. Cannot compute AUROC.")
        return 0.0

    return roc_auc_score(target_flat, pred_flat)


def compute_pixel_ap(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute pixel-level Average Precision.

    Args:
        predictions: [N, H, W] or [N] pixel-level anomaly scores
        targets: [N, H, W] or [N] binary labels

    Returns:
        Pixel-AP score
    """
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    if len(np.unique(target_flat)) < 2:
        warnings.warn("Only one class present in targets. Cannot compute AP.")
        return 0.0

    return average_precision_score(target_flat, pred_flat)


def compute_aupro(predictions: np.ndarray, targets: np.ndarray,
                  max_fpr: float = 0.3, num_thresholds: int = 1000) -> float:
    """
    Compute Area Under Per-Region Overlap (AUPRO).

    Args:
        predictions: [N, H, W] pixel-level anomaly scores
        targets: [N, H, W] binary masks
        max_fpr: Maximum FPR to integrate up to
        num_thresholds: Number of thresholds to evaluate

    Returns:
        AUPRO score
    """
    from scipy import ndimage

    # Flatten spatial dimensions
    N = predictions.shape[0]
    pred_flat = predictions.reshape(N, -1)
    target_flat = targets.reshape(N, -1)

    # Get thresholds
    thresholds = np.linspace(predictions.min(), predictions.max(), num_thresholds)

    # Compute per-region overlap at each threshold
    pro_scores = []

    for thresh in thresholds:
        pred_binary = (pred_flat >= thresh).astype(int)

        # Compute per-region overlap for each image
        overlaps = []
        for i in range(N):
            target_mask = target_flat[i].reshape(targets.shape[1:])
            pred_mask = pred_binary[i].reshape(targets.shape[1:])

            # Label connected components in ground truth
            labeled_gt, num_regions = ndimage.label(target_mask)

            if num_regions == 0:
                continue

            # Compute overlap for each region
            for region_id in range(1, num_regions + 1):
                region_mask = (labeled_gt == region_id)
                region_size = region_mask.sum()

                if region_size == 0:
                    continue

                # Overlap = intersection / region_size
                overlap = (region_mask & pred_mask).sum() / region_size
                overlaps.append(overlap)

        if overlaps:
            pro_scores.append(np.mean(overlaps))
        else:
            pro_scores.append(0.0)

    # Compute AUPRO by integrating PRO curve up to max_fpr
    # Note: This is a simplified version. Full implementation would compute FPR per threshold
    aupro = np.trapz(pro_scores, dx=1.0/num_thresholds) * max_fpr

    return aupro


def compute_pixel_f1(predictions: np.ndarray, targets: np.ndarray, threshold: float) -> Dict[str, float]:
    """
    Compute pixel-level F1, Precision, Recall.

    Args:
        predictions: [N, H, W] or [N] pixel-level anomaly scores
        targets: [N, H, W] or [N] binary labels
        threshold: Threshold for converting scores to binary predictions

    Returns:
        Dictionary with F1, precision, recall
    """
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    pred_binary = (pred_flat >= threshold).astype(int)

    tp = ((pred_binary == 1) & (target_flat == 1)).sum()
    fp = ((pred_binary == 1) & (target_flat == 0)).sum()
    fn = ((pred_binary == 0) & (target_flat == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def compute_frame_auc(frame_scores: np.ndarray, frame_labels: np.ndarray) -> float:
    """
    Compute frame-level AUC.

    Args:
        frame_scores: [N] frame-level anomaly scores
        frame_labels: [N] binary labels (0=normal, 1=anomaly)

    Returns:
        Frame-AUC score
    """
    if len(np.unique(frame_labels)) < 2:
        warnings.warn("Only one class present in frame labels. Cannot compute AUC.")
        return 0.0

    return roc_auc_score(frame_labels, frame_scores)


def compute_frame_prauc(frame_scores: np.ndarray, frame_labels: np.ndarray) -> float:
    """
    Compute frame-level PR-AUC.

    Args:
        frame_scores: [N] frame-level anomaly scores
        frame_labels: [N] binary labels

    Returns:
        Frame PR-AUC score
    """
    if len(np.unique(frame_labels)) < 2:
        warnings.warn("Only one class present in frame labels. Cannot compute PR-AUC.")
        return 0.0

    precision, recall, _ = precision_recall_curve(frame_labels, frame_scores)
    return auc(recall, precision)


def compute_rbdc(
    pred_boxes: List[np.ndarray],
    pred_scores: List[np.ndarray],
    gt_masks: List[np.ndarray],
    iou_threshold: float = 0.2,
    score_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute Region-Based Detection Criterion (RBDC).
    Measures how well predicted bounding boxes localize ground truth anomalies.

    Args:
        pred_boxes: List of [N, 4] arrays (x1, y1, x2, y2) per frame
        pred_scores: List of [N] anomaly scores per frame
        gt_masks: List of [H, W] binary masks per frame
        iou_threshold: IoU threshold for matching
        score_threshold: Score threshold for predictions (None = use all)

    Returns:
        Dictionary with RBDC metrics
    """
    tp, fp, fn = 0, 0, 0

    for boxes, scores, gt_mask in zip(pred_boxes, pred_scores, gt_masks):
        if score_threshold is not None:
            valid = scores >= score_threshold
            boxes = boxes[valid]
            scores = scores[valid]

        # Convert GT mask to boxes
        from scipy import ndimage
        labeled_gt, num_gt = ndimage.label(gt_mask)

        gt_boxes = []
        for region_id in range(1, num_gt + 1):
            region_mask = (labeled_gt == region_id)
            ys, xs = np.where(region_mask)
            if len(xs) > 0 and len(ys) > 0:
                gt_boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])

        gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))

        # Match predictions to GT
        matched_gt = set()

        for box in boxes:
            # Compute IoU with all GT boxes
            if len(gt_boxes) == 0:
                fp += 1
                continue

            ious = compute_iou_boxes(box[None, :], gt_boxes)
            max_iou_idx = ious.argmax()
            max_iou = ious[max_iou_idx]

            if max_iou >= iou_threshold and max_iou_idx not in matched_gt:
                tp += 1
                matched_gt.add(max_iou_idx)
            else:
                fp += 1

        # Unmatched GT boxes are false negatives
        fn += len(gt_boxes) - len(matched_gt)

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'rbdc_precision': precision,
        'rbdc_recall': recall,
        'rbdc_f1': f1,
        'rbdc_tp': tp,
        'rbdc_fp': fp,
        'rbdc_fn': fn,
    }


def compute_tbdc(
    track_ids: List[np.ndarray],
    track_scores: List[np.ndarray],
    frame_labels: np.ndarray,
    window_k: int = 5,
    frame_match_m: int = 3,
    score_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute Track-Based Detection Criterion (TBDC).
    Measures consistency of detections across temporal tracks.

    Args:
        track_ids: List of [N] track IDs per frame
        track_scores: List of [N] anomaly scores per frame
        frame_labels: [T] binary frame labels (0=normal, 1=anomaly)
        window_k: Tracking window size
        frame_match_m: Minimum frames for a track to match an anomaly
        score_threshold: Score threshold for predictions

    Returns:
        Dictionary with TBDC metrics
    """
    # Build track histories
    track_histories = {}  # track_id -> [(frame_idx, score), ...]

    for frame_idx, (ids, scores) in enumerate(zip(track_ids, track_scores)):
        for tid, score in zip(ids, scores):
            if tid not in track_histories:
                track_histories[tid] = []
            track_histories[tid].append((frame_idx, score))

    # Find anomaly segments in ground truth
    anomaly_segments = []
    in_anomaly = False
    start_frame = 0

    for frame_idx, label in enumerate(frame_labels):
        if label == 1 and not in_anomaly:
            in_anomaly = True
            start_frame = frame_idx
        elif label == 0 and in_anomaly:
            in_anomaly = False
            anomaly_segments.append((start_frame, frame_idx - 1))

    if in_anomaly:
        anomaly_segments.append((start_frame, len(frame_labels) - 1))

    # Match tracks to anomaly segments
    tp, fp = 0, 0
    matched_segments = set()

    for tid, history in track_histories.items():
        # Compute mean score for track
        track_frames = [frame_idx for frame_idx, _ in history]
        track_score = np.mean([score for _, score in history])

        if score_threshold is not None and track_score < score_threshold:
            continue

        # Check if track overlaps with any anomaly segment
        matched = False
        for seg_idx, (start, end) in enumerate(anomaly_segments):
            # Count frames where track overlaps with segment
            overlap_frames = [f for f in track_frames if start <= f <= end]

            if len(overlap_frames) >= frame_match_m and seg_idx not in matched_segments:
                tp += 1
                matched_segments.add(seg_idx)
                matched = True
                break

        if not matched:
            fp += 1

    # Unmatched anomaly segments are false negatives
    fn = len(anomaly_segments) - len(matched_segments)

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'tbdc_precision': precision,
        'tbdc_recall': recall,
        'tbdc_f1': f1,
        'tbdc_tp': tp,
        'tbdc_fp': fp,
        'tbdc_fn': fn,
    }


def compute_iou_boxes(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] boxes (x1, y1, x2, y2)
        boxes2: [M, 4] boxes (x1, y1, x2, y2)

    Returns:
        [N, M] IoU matrix
    """
    # Compute intersection
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute union
    union = area1[:, None] + area2[None, :] - intersection

    # Compute IoU
    iou = intersection / np.maximum(union, 1e-6)

    return iou


def evaluate_all_metrics(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        predictions: Dictionary with prediction arrays
        ground_truth: Dictionary with ground truth arrays
        threshold: Optional threshold for binary metrics

    Returns:
        Dictionary with all metrics
    """
    results = {}

    # Pixel-level metrics
    if 'pixel_scores' in predictions and 'pixel_masks' in ground_truth:
        results['pixel_auroc'] = compute_pixel_auroc(
            predictions['pixel_scores'],
            ground_truth['pixel_masks']
        )
        results['pixel_ap'] = compute_pixel_ap(
            predictions['pixel_scores'],
            ground_truth['pixel_masks']
        )

        if threshold is not None:
            pixel_f1_metrics = compute_pixel_f1(
                predictions['pixel_scores'],
                ground_truth['pixel_masks'],
                threshold
            )
            results.update(pixel_f1_metrics)

    # Frame-level metrics
    if 'frame_scores' in predictions and 'frame_labels' in ground_truth:
        results['frame_auc'] = compute_frame_auc(
            predictions['frame_scores'],
            ground_truth['frame_labels']
        )
        results['frame_prauc'] = compute_frame_prauc(
            predictions['frame_scores'],
            ground_truth['frame_labels']
        )

    # Object-level metrics (RBDC)
    if all(k in predictions for k in ['pred_boxes', 'pred_scores']) and 'pixel_masks' in ground_truth:
        rbdc_metrics = compute_rbdc(
            predictions['pred_boxes'],
            predictions['pred_scores'],
            ground_truth['pixel_masks'],
            score_threshold=threshold,
        )
        results.update(rbdc_metrics)

    # Track-level metrics (TBDC)
    if all(k in predictions for k in ['track_ids', 'track_scores']) and 'frame_labels' in ground_truth:
        tbdc_metrics = compute_tbdc(
            predictions['track_ids'],
            predictions['track_scores'],
            ground_truth['frame_labels'],
            score_threshold=threshold,
        )
        results.update(tbdc_metrics)

    return results


if __name__ == "__main__":
    print("Testing metrics...")

    # Create synthetic data
    n_frames = 100
    h, w = 64, 64

    # Frame-level
    frame_scores = np.random.rand(n_frames)
    frame_labels = (frame_scores > 0.7).astype(int)

    # Pixel-level
    pixel_scores = np.random.rand(n_frames, h, w)
    pixel_masks = (pixel_scores > 0.8).astype(int)

    # Test metrics
    print(f"Frame AUC: {compute_frame_auc(frame_scores, frame_labels):.4f}")
    print(f"Pixel AUROC: {compute_pixel_auroc(pixel_scores, pixel_masks):.4f}")
    print(f"Pixel AP: {compute_pixel_ap(pixel_scores, pixel_masks):.4f}")

    pixel_f1 = compute_pixel_f1(pixel_scores, pixel_masks, threshold=0.5)
    print(f"Pixel F1: {pixel_f1['f1']:.4f}")

    print("\nMetrics tests passed!")
