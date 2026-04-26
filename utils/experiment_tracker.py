"""
Experiment Tracker (Problem 22, 23)
Tracks experiments with metadata, hyperparameters, and model versioning.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ExperimentTracker:
    """
    Lightweight experiment tracking system.
    Saves experiment metadata, hyperparameters, metrics, and model info.
    """
    
    def __init__(self, experiment_dir: Path, experiment_name: str = None):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"exp_{timestamp}"
        self.exp_path = self.experiment_dir / self.experiment_name
        self.exp_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment record
        self.record = {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now().isoformat(),
            "status": "running",
            "hyperparameters": {},
            "metrics": {},
            "model_info": {},
            "fold_results": [],
            "notes": "",
            "duration_seconds": 0,
        }
        self.start_time = time.time()
        self._save()
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters for this experiment."""
        self.record["hyperparameters"].update(params)
        self._save()
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics with optional prefix (e.g., 'fold_1/', 'test/')."""
        for key, value in metrics.items():
            metric_key = f"{prefix}{key}" if prefix else key
            # Convert numpy types to Python native
            if hasattr(value, 'item'):
                value = value.item()
            self.record["metrics"][metric_key] = value
        self._save()
    
    def log_fold_result(self, fold_idx: int, metrics: Dict[str, float]):
        """Log results for a specific cross-validation fold."""
        fold_record = {
            "fold": fold_idx,
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        for k, v in metrics.items():
            if hasattr(v, 'item'):
                v = v.item()
            fold_record["metrics"][k] = v
        self.record["fold_results"].append(fold_record)
        self._save()
    
    def log_model_info(self, model_name: str, model_path: str = None,
                       architecture: str = None, n_params: int = None):
        """Log model metadata for versioning."""
        info = {
            "name": model_name,
            "saved_at": datetime.now().isoformat(),
        }
        if model_path:
            info["path"] = str(model_path)
        if architecture:
            info["architecture"] = architecture
        if n_params:
            info["n_parameters"] = n_params
        
        self.record["model_info"][model_name] = info
        self._save()
    
    def finish(self, notes: str = ""):
        """Mark experiment as finished and save final record."""
        self.record["status"] = "completed"
        self.record["completed_at"] = datetime.now().isoformat()
        self.record["duration_seconds"] = round(time.time() - self.start_time, 2)
        self.record["notes"] = notes
        self._save()
    
    def _save(self):
        """Save experiment record to JSON."""
        record_path = self.exp_path / "experiment.json"
        # Make everything JSON serializable
        serializable = self._make_serializable(self.record)
        with open(record_path, "w", encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, default=str)
    
    def _make_serializable(self, obj):
        """Convert numpy/torch types to Python native for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        return obj
    
    @staticmethod
    def load_experiment(experiment_path: Path) -> Dict:
        """Load a saved experiment record."""
        record_path = Path(experiment_path) / "experiment.json"
        with open(record_path, encoding='utf-8', errors='ignore') as f:
            return json.load(f)
    
    @staticmethod
    def list_experiments(experiment_dir: Path) -> list:
        """List all experiments in the directory."""
        experiments = []
        for exp_dir in sorted(Path(experiment_dir).iterdir()):
            if exp_dir.is_dir() and (exp_dir / "experiment.json").exists():
                try:
                    record = ExperimentTracker.load_experiment(exp_dir)
                    experiments.append({
                        "name": record.get("experiment_name", exp_dir.name),
                        "status": record.get("status", "unknown"),
                        "created_at": record.get("created_at", ""),
                        "duration": record.get("duration_seconds", 0),
                        "metrics": record.get("metrics", {}),
                    })
                except Exception:
                    pass
        return experiments
