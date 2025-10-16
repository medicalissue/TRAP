"""
Visualization utilities for anomaly detection results.
Includes video annotation, timelines, galleries, and HTML reports.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class AnomalyVisualizer:
    """Visualizer for anomaly detection results."""

    def __init__(self, output_dir: str = "results/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_annotated_video(
        self,
        video_path: str,
        frame_scores: np.ndarray,
        track_scores: Optional[List[Dict]] = None,
        output_path: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """
        Create video with anomaly score annotations.

        Args:
            video_path: Path to input video
            frame_scores: [T] frame-level scores
            track_scores: List of per-frame track score dicts
            output_path: Path to save output
            threshold: Anomaly threshold for coloring
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_path is None:
            output_path = self.output_dir / f"{Path(video_path).stem}_annotated.mp4"

        # Setup writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx < len(frame_scores):
                score = frame_scores[frame_idx]

                # Draw frame score bar
                frame = self.draw_score_bar(frame, score, threshold)

                # Draw track boxes if available
                if track_scores and frame_idx < len(track_scores):
                    frame = self.draw_track_boxes(frame, track_scores[frame_idx], threshold)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        print(f"Annotated video saved: {output_path}")

    def draw_score_bar(
        self,
        frame: np.ndarray,
        score: float,
        threshold: float
    ) -> np.ndarray:
        """Draw anomaly score bar on frame."""
        height, width = frame.shape[:2]

        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), (310, 60), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (310, 60), (255, 255, 255), 2)

        # Draw score bar
        bar_width = int(300 * min(score, 1.0))
        color = (0, 255, 0) if score < threshold else (0, 0, 255)
        cv2.rectangle(frame, (10, 40), (10 + bar_width, 60), color, -1)

        # Draw text
        text = f"Anomaly: {score:.3f}"
        cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw threshold line
        thresh_x = int(10 + 300 * threshold)
        cv2.line(frame, (thresh_x, 40), (thresh_x, 60), (255, 255, 0), 2)

        return frame

    def draw_track_boxes(
        self,
        frame: np.ndarray,
        track_scores: Dict[int, Dict],
        threshold: float
    ) -> np.ndarray:
        """Draw bounding boxes with anomaly scores."""
        for track_id, info in track_scores.items():
            score = info.get('anomaly_score', 0.0)
            box = info.get('box', None)

            if box is None:
                continue

            box = box.astype(int)

            # Color based on score
            if score < threshold * 0.5:
                color = (0, 255, 0)  # Green
            elif score < threshold:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red

            # Draw box
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            # Draw label
            label = f"ID{track_id}: {score:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(
                frame,
                (box[0], box[1] - label_size[1] - 10),
                (box[0] + label_size[0], box[1]),
                color, -1
            )
            cv2.putText(
                frame, label,
                (box[0], box[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2
            )

        return frame

    def plot_track_timeline(
        self,
        track_id: int,
        scores: List[float],
        frame_indices: List[int],
        threshold: float = 0.5,
        save_path: Optional[str] = None
    ):
        """
        Plot anomaly score timeline for a single track.

        Args:
            track_id: Track ID
            scores: List of anomaly scores
            frame_indices: List of frame indices
            threshold: Anomaly threshold
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        # Plot scores
        ax.plot(frame_indices, scores, 'b-', linewidth=2, label='Anomaly Score')

        # Plot threshold
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label='Threshold')

        # Highlight anomalies
        anomaly_indices = [i for i, s in enumerate(scores) if s > threshold]
        if anomaly_indices:
            ax.scatter(
                [frame_indices[i] for i in anomaly_indices],
                [scores[i] for i in anomaly_indices],
                c='r', s=50, zorder=5, label='Anomaly'
            )

        ax.set_xlabel('Frame Index', fontsize=12)
        ax.set_ylabel('Anomaly Score', fontsize=12)
        ax.set_title(f'Track {track_id} Timeline', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / f"track_{track_id}_timeline.png"

        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        print(f"Timeline saved: {save_path}")

    def plot_frame_scores(
        self,
        frame_scores: np.ndarray,
        frame_labels: Optional[np.ndarray] = None,
        threshold: float = 0.5,
        save_path: Optional[str] = None,
        top_k: int = 20
    ):
        """
        Plot frame-level anomaly scores.

        Args:
            frame_scores: [T] frame scores
            frame_labels: [T] ground truth labels (optional)
            threshold: Anomaly threshold
            save_path: Path to save plot
            top_k: Number of top anomalies to highlight
        """
        fig, ax = plt.subplots(figsize=(14, 5))

        frame_indices = np.arange(len(frame_scores))

        # Plot all scores
        ax.plot(frame_indices, frame_scores, 'b-', linewidth=1, alpha=0.7, label='Frame Score')

        # Plot threshold
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label='Threshold')

        # Highlight top-K anomalies
        top_k_indices = np.argsort(frame_scores)[-top_k:]
        ax.scatter(
            top_k_indices,
            frame_scores[top_k_indices],
            c='r', s=50, zorder=5, label=f'Top-{top_k} Anomalies'
        )

        # Plot ground truth if available
        if frame_labels is not None:
            anomaly_frames = frame_indices[frame_labels == 1]
            if len(anomaly_frames) > 0:
                for frame_idx in anomaly_frames:
                    ax.axvspan(frame_idx, frame_idx + 1, alpha=0.2, color='red')

        ax.set_xlabel('Frame Index', fontsize=12)
        ax.set_ylabel('Anomaly Score', fontsize=12)
        ax.set_title('Frame-Level Anomaly Scores', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "frame_scores.png"

        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        print(f"Frame scores plot saved: {save_path}")

    def create_anomaly_gallery(
        self,
        video_path: str,
        frame_scores: np.ndarray,
        top_k: int = 12,
        save_path: Optional[str] = None
    ):
        """
        Create gallery of top anomalous frames.

        Args:
            video_path: Path to video
            frame_scores: [T] frame scores
            top_k: Number of top frames to show
            save_path: Path to save gallery
        """
        # Get top-K frames
        top_k_indices = np.argsort(frame_scores)[-top_k:][::-1]

        # Read frames
        cap = cv2.VideoCapture(video_path)
        frames = []

        for idx in top_k_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((idx, frame, frame_scores[idx]))

        cap.release()

        # Create gallery
        n_rows = (top_k + 3) // 4
        n_cols = 4

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if top_k > 1 else [axes]

        for i, (frame_idx, frame, score) in enumerate(frames):
            axes[i].imshow(frame)
            axes[i].set_title(f"Frame {frame_idx}\nScore: {score:.3f}", fontsize=10)
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(len(frames), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "anomaly_gallery.png"

        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        print(f"Anomaly gallery saved: {save_path}")

    def create_html_report(
        self,
        results: Dict,
        metrics: Dict[str, float],
        output_path: Optional[str] = None
    ):
        """
        Create HTML summary report.

        Args:
            results: Inference results dictionary
            metrics: Evaluation metrics
            output_path: Path to save HTML
        """
        if output_path is None:
            output_path = self.output_dir / "report.html"

        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Anomaly Detection Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #4CAF50; color: white; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                .metric { font-weight: bold; }
                img { max-width: 100%; height: auto; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>Video Anomaly Detection Report</h1>
        """

        # Video info
        if 'video_info' in results:
            info = results['video_info']
            html += f"""
            <h2>Video Information</h2>
            <table>
                <tr><th>Property</th><th>Value</th></tr>
                <tr><td>Total Frames</td><td>{info.get('total_frames', 'N/A')}</td></tr>
                <tr><td>FPS</td><td>{info.get('fps', 'N/A'):.2f}</td></tr>
                <tr><td>Resolution</td><td>{info.get('width', 'N/A')}x{info.get('height', 'N/A')}</td></tr>
            </table>
            """

        # Metrics
        if metrics:
            html += """
            <h2>Performance Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
            """
            for key, value in metrics.items():
                if isinstance(value, float):
                    html += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>"
                else:
                    html += f"<tr><td>{key}</td><td>{value}</td></tr>"
            html += "</table>"

        # Anomaly statistics
        if 'frame_results' in results:
            frame_scores = [r['frame_score'] for r in results['frame_results']]
            html += f"""
            <h2>Anomaly Statistics</h2>
            <table>
                <tr><th>Statistic</th><th>Value</th></tr>
                <tr><td>Mean Score</td><td>{np.mean(frame_scores):.4f}</td></tr>
                <tr><td>Max Score</td><td>{np.max(frame_scores):.4f}</td></tr>
                <tr><td>Min Score</td><td>{np.min(frame_scores):.4f}</td></tr>
                <tr><td>Std Dev</td><td>{np.std(frame_scores):.4f}</td></tr>
            </table>
            """

        # Track statistics
        if 'track_histories' in results:
            num_tracks = len(results['track_histories'])
            html += f"""
            <h2>Track Statistics</h2>
            <table>
                <tr><th>Statistic</th><th>Value</th></tr>
                <tr><td>Total Tracks</td><td>{num_tracks}</td></tr>
            </table>
            """

        html += """
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html)

        print(f"HTML report saved: {output_path}")


if __name__ == "__main__":
    # Test visualizer
    print("Testing visualizer...")

    visualizer = AnomalyVisualizer("test_viz")

    # Create dummy data
    n_frames = 100
    frame_scores = np.random.beta(2, 5, size=n_frames)
    frame_scores[40:50] = np.random.beta(5, 2, size=10)  # Anomaly segment

    # Plot frame scores
    visualizer.plot_frame_scores(frame_scores, threshold=0.5)

    # Plot track timeline
    visualizer.plot_track_timeline(
        track_id=1,
        scores=frame_scores[:50].tolist(),
        frame_indices=list(range(50)),
        threshold=0.5
    )

    # Create HTML report
    results = {
        'video_info': {'total_frames': n_frames, 'fps': 30.0, 'width': 640, 'height': 480},
        'frame_results': [{'frame_score': s} for s in frame_scores],
        'track_histories': {i: {} for i in range(10)},
    }

    metrics = {
        'frame_auc': 0.95,
        'pixel_auroc': 0.92,
        'rbdc_f1': 0.88,
    }

    visualizer.create_html_report(results, metrics)

    print("Visualizer tests passed!")
