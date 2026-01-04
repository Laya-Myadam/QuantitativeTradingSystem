"""
MLOps Monitoring and Drift Detection System
Tracks model performance, detects drift, triggers retraining
"""
import random
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import deque
import math

class ModelMonitor:
    """
    Monitor model performance in real-time
    Tracks predictions, actuals, and calculates metrics
    """
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size

        # Store recent predictions and actuals
        self.predictions = deque(maxlen=window_size)
        self.actuals = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

        # Performance metrics over time
        self.metrics_history = []

        # Baseline metrics (set during initial training)
        self.baseline_rmse = 5.0
        self.baseline_mae = 3.5

        # Alert thresholds
        self.thresholds = {
            'rmse_increase': 0.25,  # 25% increase triggers alert
            'mae_increase': 0.25,
            'accuracy_drop': 0.15
        }

        # Alert log
        self.alerts = []

    def log_prediction(self, prediction: float, actual: Optional[float] = None):
        """
        Log a prediction and optionally its actual value
        """
        timestamp = datetime.now()

        self.predictions.append(prediction)
        self.timestamps.append(timestamp)

        if actual is not None:
            self.actuals.append(actual)

            # Calculate metrics if we have enough data
            if len(self.actuals) >= 50:
                metrics = self._calculate_metrics()
                self.metrics_history.append({
                    'timestamp': timestamp.isoformat(),
                    **metrics
                })

                # Check for performance degradation
                alerts = self._check_alerts(metrics)
                if alerts:
                    self.alerts.extend(alerts)

    def _calculate_metrics(self) -> Dict:
        """Calculate current performance metrics"""
        if len(self.actuals) < 10:
            return {}

        preds = list(self.predictions)[-len(self.actuals):]
        actuals = list(self.actuals)

        # RMSE
        squared_errors = [(p - a) ** 2 for p, a in zip(preds, actuals)]
        rmse = math.sqrt(sum(squared_errors) / len(squared_errors))

        # MAE
        abs_errors = [abs(p - a) for p, a in zip(preds, actuals)]
        mae = sum(abs_errors) / len(abs_errors)

        # MAPE
        mape = sum(abs(a - p) / abs(a) * 100 for p, a in zip(preds, actuals) if a != 0) / len(actuals)

        # Direction accuracy (did we predict up/down correctly?)
        if len(actuals) > 1:
            direction_correct = sum(
                1 for i in range(1, len(actuals))
                if (preds[i] > preds[i-1]) == (actuals[i] > actuals[i-1])
            )
            direction_accuracy = (direction_correct / (len(actuals) - 1)) * 100
        else:
            direction_accuracy = 0

        return {
            'rmse': round(rmse, 3),
            'mae': round(mae, 3),
            'mape': round(mape, 2),
            'direction_accuracy': round(direction_accuracy, 1),
            'n_samples': len(actuals)
        }

    def _check_alerts(self, current_metrics: Dict) -> List[Dict]:
        """Check if metrics trigger any alerts"""
        alerts = []

        # Check RMSE increase
        rmse_increase = (current_metrics['rmse'] - self.baseline_rmse) / self.baseline_rmse
        if rmse_increase > self.thresholds['rmse_increase']:
            alerts.append({
                'type': 'PERFORMANCE_DEGRADATION',
                'severity': 'HIGH',
                'metric': 'RMSE',
                'message': f"RMSE increased by {rmse_increase*100:.1f}%",
                'current_value': current_metrics['rmse'],
                'baseline_value': self.baseline_rmse,
                'timestamp': datetime.now().isoformat(),
                'recommendation': 'Model retraining recommended'
            })

        # Check MAE increase
        mae_increase = (current_metrics['mae'] - self.baseline_mae) / self.baseline_mae
        if mae_increase > self.thresholds['mae_increase']:
            alerts.append({
                'type': 'PERFORMANCE_DEGRADATION',
                'severity': 'MEDIUM',
                'metric': 'MAE',
                'message': f"MAE increased by {mae_increase*100:.1f}%",
                'current_value': current_metrics['mae'],
                'baseline_value': self.baseline_mae,
                'timestamp': datetime.now().isoformat(),
                'recommendation': 'Monitor closely, consider retraining'
            })

        return alerts

    def get_summary(self) -> Dict:
        """Get monitoring summary"""
        if not self.metrics_history:
            return {
                'status': 'initializing',
                'message': 'Not enough data yet'
            }

        current_metrics = self.metrics_history[-1]

        # Determine health status
        rmse_increase = (current_metrics['rmse'] - self.baseline_rmse) / self.baseline_rmse

        if rmse_increase > 0.4:
            status = 'critical'
            message = 'Model performance severely degraded'
        elif rmse_increase > 0.25:
            status = 'warning'
            message = 'Model performance declining'
        else:
            status = 'healthy'
            message = 'Model performing well'

        return {
            'status': status,
            'message': message,
            'current_metrics': current_metrics,
            'baseline_metrics': {
                'rmse': self.baseline_rmse,
                'mae': self.baseline_mae
            },
            'total_predictions': len(self.predictions),
            'recent_alerts': self.alerts[-5:] if self.alerts else [],
            'monitoring_since': self.timestamps[0].isoformat() if self.timestamps else None
        }


class DriftDetector:
    """
    Detect statistical drift in model inputs/outputs
    Uses multiple drift detection methods
    """
    def __init__(self, reference_window: int = 1000, detection_window: int = 100):
        self.reference_window = reference_window
        self.detection_window = detection_window

        # Store reference distribution
        self.reference_data = deque(maxlen=reference_window)
        self.current_data = deque(maxlen=detection_window)

        # Drift detection history
        self.drift_history = []

    def add_reference_data(self, values: List[float]):
        """Add data to reference distribution"""
        self.reference_data.extend(values)

    def add_current_data(self, value: float):
        """Add current data point"""
        self.current_data.append(value)

    def check_drift(self) -> Dict:
        """
        Check for distribution drift
        Returns drift status and metrics
        """
        if len(self.reference_data) < 50 or len(self.current_data) < 20:
            return {
                'detected': False,
                'confidence': 0,
                'psi_score': 0,
                'mean_shift': 0,
                'std_shift': 0,
                'method': 'insufficient_data',
                'severity': 'LOW',
                'message': 'Not enough data for drift detection',
                'recommendation': 'Collecting data...',
                'timestamp': datetime.now().isoformat()
            }

        try:
            ref_data = list(self.reference_data)
            curr_data = list(self.current_data)

            # Calculate distribution statistics
            ref_mean = sum(ref_data) / len(ref_data)
            ref_std = math.sqrt(sum((x - ref_mean) ** 2 for x in ref_data) / len(ref_data))

            curr_mean = sum(curr_data) / len(curr_data)
            curr_std = math.sqrt(sum((x - curr_mean) ** 2 for x in curr_data) / len(curr_data))

            # Simple drift detection: check if mean/std shifted significantly
            mean_shift = abs(curr_mean - ref_mean) / ref_std if ref_std > 0 else 0
            std_shift = abs(curr_std - ref_std) / ref_std if ref_std > 0 else 0

            # Population Stability Index (PSI)
            psi_score = self._calculate_psi(ref_data, curr_data)

            # Determine if drift detected
            drift_detected = psi_score > 0.2 or mean_shift > 2 or std_shift > 0.5

            # Calculate confidence
            confidence = min(100, (psi_score + mean_shift + std_shift) * 20)

            result = {
                'detected': drift_detected,
                'confidence': round(confidence, 1),
                'psi_score': round(psi_score, 3),
                'mean_shift': round(mean_shift, 3),
                'std_shift': round(std_shift, 3),
                'method': 'PSI + Statistical Tests',
                'severity': self._get_severity(psi_score, mean_shift),
                'recommendation': self._get_recommendation(drift_detected, psi_score),
                'timestamp': datetime.now().isoformat()
            }

            self.drift_history.append(result)
            return result

        except Exception as e:
            # Return safe default if anything fails
            return {
                'detected': False,
                'confidence': 0,
                'psi_score': 0,
                'mean_shift': 0,
                'std_shift': 0,
                'method': 'error',
                'severity': 'LOW',
                'message': f'Error in drift detection: {str(e)}',
                'recommendation': 'Check system logs',
                'timestamp': datetime.now().isoformat()
            }

    def _calculate_psi(self, reference: List[float], current: List[float], buckets: int = 10) -> float:
        """Calculate Population Stability Index"""
        if len(reference) < buckets or len(current) < buckets:
            return 0.0  # Not enough data

        try:
            # Create buckets based on reference distribution
            ref_sorted = sorted(reference)

            # Calculate bucket edges more safely
            bucket_edges = [float('-inf')]  # Start with -inf

            for i in range(1, buckets):
                # Calculate index safely
                idx = int(i * len(ref_sorted) / buckets)
                # Make sure we don't go out of bounds
                idx = min(idx, len(ref_sorted) - 1)
                bucket_edges.append(ref_sorted[idx])

            bucket_edges.append(float('inf'))  # End with inf

            # Count samples in each bucket
            ref_counts = [0] * buckets
            curr_counts = [0] * buckets

            for val in reference:
                for i in range(buckets):
                    if bucket_edges[i] <= val < bucket_edges[i + 1]:
                        ref_counts[i] += 1
                        break

            for val in current:
                for i in range(buckets):
                    if bucket_edges[i] <= val < bucket_edges[i + 1]:
                        curr_counts[i] += 1
                        break

            # Calculate PSI
            psi = 0
            for i in range(buckets):
                ref_pct = (ref_counts[i] + 0.0001) / len(reference)
                curr_pct = (curr_counts[i] + 0.0001) / len(current)
                psi += (curr_pct - ref_pct) * math.log(curr_pct / ref_pct)

            return abs(psi)  # Return absolute value

        except Exception as e:
            print(f"PSI calculation error: {e}")
            return 0.0

    def _get_severity(self, psi_score: float, mean_shift: float) -> str:
        """Determine drift severity"""
        if psi_score > 0.25 or mean_shift > 3:
            return 'CRITICAL'
        elif psi_score > 0.2 or mean_shift > 2:
            return 'HIGH'
        elif psi_score > 0.1 or mean_shift > 1:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _get_recommendation(self, drift_detected: bool, psi_score: float) -> str:
        """Get recommendation based on drift"""
        if not drift_detected:
            return "No action needed. Continue monitoring."

        if psi_score > 0.25:
            return "URGENT: Immediate model retraining required."
        elif psi_score > 0.2:
            return "WARNING: Schedule model retraining soon."
        else:
            return "CAUTION: Monitor closely for continued drift."

    def get_drift_report(self) -> Dict:
        """Get comprehensive drift report"""
        if not self.drift_history:
            return {
                'status': 'no_checks_performed',
                'message': 'No drift checks have been performed yet'
            }

        recent_checks = self.drift_history[-20:]
        drift_detected_count = sum(1 for c in recent_checks if c['detected'])

        return {
            'total_checks': len(self.drift_history),
            'recent_drift_rate': round((drift_detected_count / len(recent_checks)) * 100, 1),
            'last_check': recent_checks[-1],
            'avg_psi_score': round(sum(c['psi_score'] for c in recent_checks) / len(recent_checks), 3),
            'status': 'DRIFT_DETECTED' if recent_checks[-1]['detected'] else 'STABLE'
        }


class AutoRetrainScheduler:
    """
    Automatically schedule and trigger model retraining
    """
    def __init__(self):
        self.last_retrain = datetime.now()
        self.retrain_interval = timedelta(days=7)  # Retrain weekly
        self.retrain_history = []

        # Retraining triggers
        self.triggers = {
            'drift_detected': True,
            'performance_degradation': True,
            'scheduled_interval': True
        }

    def should_retrain(self, drift_status: Dict, monitor_status: Dict) -> tuple[bool, str]:
        """
        Determine if retraining should be triggered
        Returns (should_retrain, reason)
        """
        reasons = []

        # Check drift
        if self.triggers['drift_detected'] and drift_status.get('detected'):
            reasons.append(f"Drift detected (PSI: {drift_status.get('psi_score', 0):.3f})")

        # Check performance
        if self.triggers['performance_degradation']:
            status = monitor_status.get('status')
            if status in ['warning', 'critical']:
                reasons.append(f"Performance degraded ({status})")

        # Check scheduled interval
        if self.triggers['scheduled_interval']:
            time_since_retrain = datetime.now() - self.last_retrain
            if time_since_retrain >= self.retrain_interval:
                reasons.append("Scheduled retraining interval reached")

        should_retrain = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "No retraining needed"

        return should_retrain, reason

    def trigger_retrain(self, reason: str) -> Dict:
        """
        Trigger model retraining
        """
        retrain_record = {
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'status': 'initiated',
            'duration_seconds': 0
        }

        # Simulate retraining
        import time
        start = time.time()
        time.sleep(0.5)  # Simulate work
        duration = time.time() - start

        retrain_record['duration_seconds'] = round(duration, 2)
        retrain_record['status'] = 'completed'

        self.retrain_history.append(retrain_record)
        self.last_retrain = datetime.now()

        return retrain_record

    def get_retrain_schedule(self) -> Dict:
        """Get retraining schedule info"""
        time_since_last = datetime.now() - self.last_retrain
        time_until_next = self.retrain_interval - time_since_last

        return {
            'last_retrain': self.last_retrain.isoformat(),
            'time_since_last': str(time_since_last),
            'next_scheduled': (self.last_retrain + self.retrain_interval).isoformat(),
            'time_until_next': str(max(time_until_next, timedelta(0))),
            'total_retrains': len(self.retrain_history),
            'recent_retrains': self.retrain_history[-5:]
        }


# Global instances
monitor = ModelMonitor()
drift_detector = DriftDetector()
retrain_scheduler = AutoRetrainScheduler()


def simulate_monitoring_data():
    """Generate simulated monitoring data for demo"""
    try:
        # Add more reference data
        reference = [random.gauss(150, 10) for _ in range(200)]
        drift_detector.add_reference_data(reference)

        # Add current data (with slight drift)
        for _ in range(100):
            value = random.gauss(155, 12)  # Slightly shifted
            drift_detector.add_current_data(value)
            monitor.log_prediction(value, value + random.gauss(0, 2))
    except Exception as e:
        print(f"Error in simulate_monitoring_data: {e}")


def get_mlops_dashboard() -> Dict:
    """Get complete MLOps dashboard data"""
    drift_status = drift_detector.check_drift()
    monitor_status = monitor.get_summary()
    should_retrain, reason = retrain_scheduler.should_retrain(drift_status, monitor_status)

    return {
        'monitoring': monitor_status,
        'drift_detection': drift_status,
        'drift_report': drift_detector.get_drift_report(),
        'retraining': {
            'should_retrain': should_retrain,
            'reason': reason,
            'schedule': retrain_scheduler.get_retrain_schedule()
        },
        'system_health': {
            'status': 'healthy' if not should_retrain else 'needs_attention',
            'uptime': '99.9%',
            'last_updated': datetime.now().isoformat()
        }
    }


# Initialize with some demo data
simulate_monitoring_data()