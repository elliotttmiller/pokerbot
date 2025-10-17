#!/usr/bin/env python3
"""
Real-Time Progress Tracking and Monitoring System

Provides comprehensive tracking for data generation, training, and validation with:
- Real-time progress streaming
- Performance metrics analysis
- Bottleneck detection
- Automated recommendations
- Web dashboard support

Usage:
  # Track data generation
  python scripts/track_progress.py --task generation --config config.json
  
  # Track training
  python scripts/track_progress.py --task training --model-dir models/
  
  # Launch dashboard
  python scripts/track_progress.py --dashboard --port 8080
"""

import sys
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
pythonpath = os.environ.get("PYTHONPATH")
if pythonpath:
    for p in pythonpath.split(os.pathsep):
        if p and p not in sys.path:
            sys.path.insert(0, p)
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import json
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProgressTracker:
    """Real-time progress tracking with analytics and recommendations."""
    
    def __init__(self, task_type: str, config: Dict):
        self.task_type = task_type
        self.config = config
        self.start_time = time.time()
        self.metrics = {
            'progress': 0.0,
            'eta': None,
            'speed': 0.0,
            'status': 'initializing',
            'bottlenecks': [],
            'recommendations': []
        }
        self.history = []
        self.update_queue = queue.Queue()
        
    def update(self, progress: float, **kwargs):
        """Update progress with real-time metrics."""
        elapsed = time.time() - self.start_time
        
        self.metrics['progress'] = progress
        self.metrics['elapsed_time'] = elapsed
        
        if progress > 0:
            total_time = elapsed / progress
            self.metrics['eta'] = total_time - elapsed
            self.metrics['speed'] = progress / elapsed
        
        # Update additional metrics
        for key, value in kwargs.items():
            self.metrics[key] = value
        
        # Store in history
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'progress': progress,
            'metrics': self.metrics.copy()
        })
        
        # Analyze for bottlenecks
        self._analyze_bottlenecks()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Queue update for dashboard
        self.update_queue.put(self.metrics.copy())
        
    def _analyze_bottlenecks(self):
        """Detect performance bottlenecks using real-time analysis."""
        bottlenecks = []
        
        # Check if speed is degrading
        if len(self.history) > 10:
            recent_speeds = [h['metrics'].get('speed', 0) for h in self.history[-10:]]
            if len(recent_speeds) > 1:
                avg_speed = sum(recent_speeds) / len(recent_speeds)
                latest_speed = recent_speeds[-1]
                
                if latest_speed < avg_speed * 0.8:
                    bottlenecks.append({
                        'type': 'speed_degradation',
                        'severity': 'high',
                        'message': f'Speed degraded by {(1 - latest_speed/avg_speed)*100:.1f}%',
                        'suggestion': 'Check system resources (CPU, memory, disk I/O)'
                    })
        
        # Check memory usage if available
        if 'memory_usage' in self.metrics:
            mem_pct = self.metrics['memory_usage']
            if mem_pct > 90:
                bottlenecks.append({
                    'type': 'high_memory',
                    'severity': 'critical',
                    'message': f'Memory usage at {mem_pct:.1f}%',
                    'suggestion': 'Reduce batch size or enable gradient accumulation'
                })
            elif mem_pct > 80:
                bottlenecks.append({
                    'type': 'high_memory',
                    'severity': 'warning',
                    'message': f'Memory usage at {mem_pct:.1f}%',
                    'suggestion': 'Monitor closely, consider reducing batch size'
                })
        
        self.metrics['bottlenecks'] = bottlenecks
        
    def _generate_recommendations(self):
        """Generate intelligent recommendations based on metrics."""
        recommendations = []
        
        progress = self.metrics['progress']
        elapsed = self.metrics['elapsed_time']
        
        # Early-stage recommendations
        if progress < 0.1 and elapsed > 300:  # 5 minutes
            speed = self.metrics.get('speed', 0)
            if speed > 0:
                estimated_total = 1.0 / speed
                if estimated_total > 86400:  # >1 day
                    recommendations.append({
                        'priority': 'high',
                        'category': 'optimization',
                        'message': f'Estimated total time: {estimated_total/3600:.1f} hours',
                        'suggestions': [
                            'Consider using adaptive CFR for 20-30% speedup',
                            'Enable multi-GPU training if available',
                            'Use distributed data generation across multiple machines'
                        ]
                    })
        
        # Mid-stage recommendations
        if 0.3 <= progress <= 0.7:
            # Check if we're on track
            if 'eta' in self.metrics and self.metrics['eta']:
                eta_hours = self.metrics['eta'] / 3600
                if eta_hours > 48:  # >2 days remaining
                    recommendations.append({
                        'priority': 'medium',
                        'category': 'efficiency',
                        'message': f'ETA: {eta_hours:.1f} hours remaining',
                        'suggestions': [
                            'Consider checkpointing and resuming with more resources',
                            'Parallelize if not already doing so'
                        ]
                    })
        
        # Training-specific recommendations
        if self.task_type == 'training':
            if 'loss' in self.metrics:
                loss_history = [h['metrics'].get('loss', float('inf')) for h in self.history[-20:]]
                if len(loss_history) > 10:
                    # Check for plateau
                    recent_loss = sum(loss_history[-5:]) / 5
                    older_loss = sum(loss_history[-10:-5]) / 5
                    
                    if abs(recent_loss - older_loss) / older_loss < 0.01:
                        recommendations.append({
                            'priority': 'high',
                            'category': 'training',
                            'message': 'Loss plateau detected',
                            'suggestions': [
                                'Reduce learning rate by 50%',
                                'Enable early stopping if not already active',
                                'Check if model has converged'
                            ]
                        })
        
        # Data generation recommendations
        if self.task_type == 'generation':
            if 'samples_generated' in self.metrics:
                samples = self.metrics['samples_generated']
                if samples < 10000 and progress > 0.8:
                    recommendations.append({
                        'priority': 'critical',
                        'category': 'data_quality',
                        'message': f'Only {samples} samples generated',
                        'suggestions': [
                            'This is insufficient for production use',
                            'Minimum recommended: 50,000 samples',
                            'Production: 100,000+ samples',
                            'Championship: 500,000+ samples'
                        ]
                    })
        
        self.metrics['recommendations'] = recommendations
    
    def get_status(self) -> Dict:
        """Get current status with all metrics."""
        return {
            'task_type': self.task_type,
            'metrics': self.metrics,
            'history': self.history[-100:],  # Last 100 entries
            'timestamp': datetime.now().isoformat()
        }
    
    def save_checkpoint(self, filepath: str):
        """Save tracking state for resumption."""
        checkpoint = {
            'task_type': self.task_type,
            'config': self.config,
            'metrics': self.metrics,
            'history': self.history,
            'start_time': self.start_time
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Progress checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Resume from saved checkpoint."""
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        self.task_type = checkpoint['task_type']
        self.config = checkpoint['config']
        self.metrics = checkpoint['metrics']
        self.history = checkpoint['history']
        self.start_time = checkpoint['start_time']
        
        logger.info(f"Progress resumed from checkpoint {filepath}")


class DashboardServer:
    """Lightweight web dashboard for progress monitoring."""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.tracker = None
        
    def start(self, tracker: ProgressTracker):
        """Start dashboard server."""
        self.tracker = tracker
        
        try:
            import http.server
            import socketserver
            
            class DashboardHandler(http.server.SimpleHTTPRequestHandler):
                def do_GET(self_):
                    if self_.path == '/api/status':
                        self_._serve_status()
                    elif self_.path == '/':
                        self_._serve_dashboard()
                    else:
                        super().do_GET()
                
                def _serve_status(self_):
                    """Serve current status as JSON."""
                    self_.send_response(200)
                    self_.send_header('Content-type', 'application/json')
                    self_.send_header('Access-Control-Allow-Origin', '*')
                    self_.end_headers()
                    
                    status = self.tracker.get_status()
                    self_.wfile.write(json.dumps(status).encode())
                
                def _serve_dashboard(self_):
                    """Serve dashboard HTML."""
                    self_.send_response(200)
                    self_.send_header('Content-type', 'text/html')
                    self_.end_headers()
                    
                    html = self._generate_dashboard_html()
                    self_.wfile.write(html.encode())
                
                def log_message(self_, format, *args):
                    """Suppress request logging."""
                    pass
            
            with socketserver.TCPServer(("", self.port), DashboardHandler) as httpd:
                logger.info(f"Dashboard server started at http://localhost:{self.port}")
                logger.info("Press Ctrl+C to stop")
                httpd.serve_forever()
                
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
    
    def _generate_dashboard_html(self) -> str:
        """Generate real-time dashboard HTML."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Training Progress Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #0a0e27;
            color: #e0e6ed;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            font-size: 2rem;
            margin-bottom: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: #1a1f3a;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .card h2 {
            font-size: 1.2rem;
            margin-bottom: 16px;
            color: #8b95a8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric {
            font-size: 2.5rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
        }
        .label {
            font-size: 0.9rem;
            color: #8b95a8;
        }
        .progress-bar {
            width: 100%;
            height: 32px;
            background: #0f1729;
            border-radius: 16px;
            overflow: hidden;
            margin: 16px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .status {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .status.running { background: #10b981; color: white; }
        .status.warning { background: #f59e0b; color: white; }
        .status.error { background: #ef4444; color: white; }
        .recommendation {
            background: #252d4d;
            border-left: 4px solid #667eea;
            padding: 12px 16px;
            margin-bottom: 12px;
            border-radius: 6px;
        }
        .recommendation.high { border-left-color: #ef4444; }
        .recommendation.medium { border-left-color: #f59e0b; }
        .recommendation h3 {
            font-size: 0.95rem;
            margin-bottom: 8px;
            color: #e0e6ed;
        }
        .recommendation ul {
            list-style: none;
            font-size: 0.9rem;
            color: #8b95a8;
        }
        .recommendation li:before {
            content: "→ ";
            color: #667eea;
            font-weight: bold;
        }
        .bottleneck {
            background: #3d1f1f;
            border-left: 4px solid #ef4444;
            padding: 12px 16px;
            margin-bottom: 12px;
            border-radius: 6px;
        }
        .chart-container {
            background: #0f1729;
            padding: 16px;
            border-radius: 8px;
            margin-top: 16px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Training Progress Dashboard</h1>
        
        <div class="grid">
            <div class="card">
                <h2>Progress</h2>
                <div class="metric" id="progress">0%</div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-fill" style="width: 0%">0%</div>
                </div>
                <div class="label">Status: <span class="status running" id="status">Initializing</span></div>
            </div>
            
            <div class="card">
                <h2>Time</h2>
                <div class="metric" id="elapsed">00:00:00</div>
                <div class="label">Elapsed</div>
                <div class="metric" id="eta" style="font-size: 1.5rem; margin-top: 12px;">Calculating...</div>
                <div class="label">Estimated Remaining</div>
            </div>
            
            <div class="card">
                <h2>Performance</h2>
                <div class="metric" id="speed">0.0</div>
                <div class="label" id="speed-unit">samples/sec</div>
                <div class="metric" id="throughput" style="font-size: 1.5rem; margin-top: 12px;">0</div>
                <div class="label">Total Processed</div>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h2>⚠️ Bottlenecks</h2>
                <div id="bottlenecks">
                    <p style="color: #8b95a8;">No bottlenecks detected</p>
                </div>
            </div>
            
            <div class="card">
                <h2>💡 Recommendations</h2>
                <div id="recommendations">
                    <p style="color: #8b95a8;">Analyzing...</p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function formatTime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }
        
        function updateDashboard() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const metrics = data.metrics;
                    
                    // Update progress
                    const progress = (metrics.progress * 100).toFixed(1);
                    document.getElementById('progress').textContent = progress + '%';
                    document.getElementById('progress-fill').style.width = progress + '%';
                    document.getElementById('progress-fill').textContent = progress + '%';
                    
                    // Update time
                    document.getElementById('elapsed').textContent = formatTime(metrics.elapsed_time || 0);
                    if (metrics.eta) {
                        document.getElementById('eta').textContent = formatTime(metrics.eta);
                    }
                    
                    // Update speed
                    const speed = (metrics.speed || 0).toFixed(2);
                    document.getElementById('speed').textContent = speed;
                    
                    // Update status
                    const statusEl = document.getElementById('status');
                    statusEl.textContent = metrics.status || 'Running';
                    statusEl.className = 'status ' + (metrics.status === 'error' ? 'error' : 'running');
                    
                    // Update bottlenecks
                    const bottlenecksEl = document.getElementById('bottlenecks');
                    if (metrics.bottlenecks && metrics.bottlenecks.length > 0) {
                        bottlenecksEl.innerHTML = metrics.bottlenecks.map(b => `
                            <div class="bottleneck">
                                <h3>${b.message}</h3>
                                <p style="margin-top: 8px; font-size: 0.9rem;">${b.suggestion}</p>
                            </div>
                        `).join('');
                    } else {
                        bottlenecksEl.innerHTML = '<p style="color: #10b981;">✓ No bottlenecks detected</p>';
                    }
                    
                    // Update recommendations
                    const recEl = document.getElementById('recommendations');
                    if (metrics.recommendations && metrics.recommendations.length > 0) {
                        recEl.innerHTML = metrics.recommendations.map(r => `
                            <div class="recommendation ${r.priority}">
                                <h3>${r.message}</h3>
                                <ul>
                                    ${r.suggestions.map(s => `<li>${s}</li>`).join('')}
                                </ul>
                            </div>
                        `).join('');
                    } else {
                        recEl.innerHTML = '<p style="color: #8b95a8;">No recommendations at this time</p>';
                    }
                })
                .catch(error => {
                    console.error('Update failed:', error);
                });
        }
        
        // Update every 2 seconds
        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</body>
</html>
        """


def main():
    parser = argparse.ArgumentParser(description='Real-time progress tracking and monitoring')
    parser.add_argument('--task', type=str, choices=['generation', 'training', 'validation'],
                       help='Task type to track')
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--dashboard', action='store_true', help='Launch web dashboard')
    parser.add_argument('--port', type=int, default=8080, help='Dashboard port')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file for resumption')
    
    args = parser.parse_args()
    
    if args.dashboard:
        # Launch dashboard server
        config = {}
        if args.config:
            with open(args.config, 'r') as f:
                config = json.load(f)
        
        tracker = ProgressTracker(args.task or 'general', config)
        
        if args.checkpoint and os.path.exists(args.checkpoint):
            tracker.load_checkpoint(args.checkpoint)
        
        dashboard = DashboardServer(args.port)
        dashboard.start(tracker)
    else:
        print("Use --dashboard to launch the monitoring dashboard")
        print(f"Example: python {sys.argv[0]} --dashboard --port 8080")


if __name__ == '__main__':
    main()
