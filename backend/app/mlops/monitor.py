"""
System Monitor
Add this in Phase 3
"""
from datetime import datetime


class SystemMonitor:
    """Monitor system performance"""

    def __init__(self):
        self.metrics = []

    def log_metric(self, name, value):
        """Log a metric"""
        self.metrics.append({
            "name": name,
            "value": value,
            "timestamp": datetime.now().isoformat()
        })

    def get_health(self):
        """Get system health"""
        return {
            "status": "healthy",
            "uptime": "99.9%",
            "metrics_logged": len(self.metrics)
        }


monitor = SystemMonitor()