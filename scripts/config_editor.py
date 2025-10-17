#!/usr/bin/env python3
"""
Web-Based Configuration Editor

Interactive web interface for managing training configurations:
- Visual configuration editing
- Profile management
- Validation and recommendations
- Export/import configurations
- Real-time preview

Usage:
  python scripts/config_editor.py --port 8090
"""

import sys
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import http.server
import socketserver
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigEditorServer:
    """Web-based configuration editor with validation."""
    
    def __init__(self, port: int = 8090):
        self.port = port
        self.config_dir = Path(__file__).parent.parent / 'config'
        
    def start(self):
        """Start the configuration editor server."""
        
        class EditorHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self_):
                if self_.path == '/':
                    self_._serve_editor()
                elif self_.path == '/api/configs':
                    self_._serve_configs_list()
                elif self_.path.startswith('/api/config/'):
                    self_._serve_config()
                else:
                    self_.send_error(404)
            
            def do_POST(self_):
                if self_.path == '/api/save':
                    self_._save_config()
                elif self_.path == '/api/validate':
                    self_._validate_config()
                else:
                    self_.send_error(404)
            
            def _serve_editor(self_):
                """Serve the main editor interface."""
                self_.send_response(200)
                self_.send_header('Content-type', 'text/html')
                self_.end_headers()
                
                # Use the helper defined in the outer scope
                html = _generate_editor_html()
                self_.wfile.write(html.encode())
            
            def _serve_configs_list(self_):
                """List available configurations."""
                configs = {
                    'data_generation': [],
                    'training': []
                }
                
                for category in ['data_generation', 'training']:
                    config_path = self.config_dir / category
                    if config_path.exists():
                        for file in config_path.glob('*.json'):
                            configs[category].append(file.stem)
                
                self_._send_json(configs)
            
            def _serve_config(self_):
                """Serve a specific configuration."""
                # Parse path: /api/config/{category}/{name}
                parts = self_.path.split('/')
                if len(parts) >= 5:
                    category = parts[3]
                    name = parts[4]
                    
                    config_file = self.config_dir / category / f"{name}.json"
                    if config_file.exists():
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        self_._send_json(config)
                    else:
                        self_.send_error(404)
                else:
                    self_.send_error(400)
            
            def _save_config(self_):
                """Save a configuration."""
                content_length = int(self_.headers['Content-Length'])
                post_data = self_.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                category = data.get('category')
                name = data.get('name')
                config = data.get('config')
                
                if not all([category, name, config]):
                    self_.send_error(400)
                    return
                
                config_file = self.config_dir / category / f"{name}.json"
                config_file.parent.mkdir(parents=True, exist_ok=True)
                
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
                self_._send_json({'status': 'success', 'message': f'Saved {category}/{name}.json'})
            
            def _validate_config(self_):
                """Validate a configuration."""
                content_length = int(self_.headers['Content-Length'])
                post_data = self_.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                config = data.get('config')
                category = data.get('category')
                
                # Call the handler's own validation method
                validation = self_._validate_configuration(config, category)
                self_._send_json(validation)
            
            def _send_json(self_, data):
                """Send JSON response."""
                self_.send_response(200)
                self_.send_header('Content-type', 'application/json')
                self_.send_header('Access-Control-Allow-Origin', '*')
                self_.end_headers()
                self_.wfile.write(json.dumps(data).encode())
            
            def _validate_configuration(self_, config: Dict, category: str) -> Dict:
                """Validate configuration and return issues/recommendations."""
                issues = []
                recommendations = []
                
                if category == 'training':
                    # Validate training config
                    if 'lr' in config:
                        lr = config['lr']
                        if lr > 0.01:
                            issues.append("Learning rate too high (>0.01)")
                        elif lr < 1e-6:
                            issues.append("Learning rate too low (<1e-6)")
                    
                    if 'batch_size' in config:
                        bs = config['batch_size']
                        if bs < 64:
                            issues.append("Batch size too small (<64)")
                            recommendations.append("Use batch size >= 512 for stable training")
                    
                    if 'epochs' in config:
                        epochs = config['epochs']
                        if epochs < 50:
                            recommendations.append("Consider training for at least 100 epochs")
                    
                    if 'use_street_weighting' in config and config['use_street_weighting']:
                        if 'street_weights' not in config:
                            issues.append("Street weighting enabled but weights not specified")
                
                elif category == 'data_generation':
                    # Validate data generation config
                    if 'samples' in config:
                        samples = config['samples']
                        if samples < 10000:
                            issues.append(f"Too few samples ({samples} < 10,000)")
                            recommendations.append("Generate at least 50,000 samples for production")
                        elif samples < 50000:
                            recommendations.append("For production, use 100,000+ samples")
                    
                    if 'cfr_iterations' in config:
                        cfr_iters = config['cfr_iterations']
                        if cfr_iters < 1500:
                            issues.append(f"CFR iterations too low ({cfr_iters} < 1500)")
                        elif cfr_iters < 2000:
                            recommendations.append("Use 2000+ iterations for better quality")
                
                return {
                    'valid': len(issues) == 0,
                    'issues': issues,
                    'recommendations': recommendations
                }
            
            def log_message(self_, format, *args):
                """Suppress request logging."""
                pass
        
        def _generate_editor_html() -> str:
            """Generate editor HTML interface."""
            return """
<!DOCTYPE html>
<html>
<head>
    <title>Configuration Editor</title>
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
        .sidebar {
            position: fixed;
            left: 20px;
            top: 80px;
            width: 250px;
            background: #1a1f3a;
            border-radius: 12px;
            padding: 20px;
            max-height: calc(100vh - 100px);
            overflow-y: auto;
        }
        .main-content {
            margin-left: 290px;
            background: #1a1f3a;
            border-radius: 12px;
            padding: 30px;
        }
        .config-list {
            margin-top: 20px;
        }
        .config-item {
            padding: 10px 15px;
            margin-bottom: 8px;
            background: #252d4d;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .config-item:hover {
            background: #2d355a;
            transform: translateX(5px);
        }
        .config-item.active {
            background: #667eea;
            color: white;
        }
        .form-group {
            margin-bottom: 20px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #8b95a8;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .form-group input, .form-group textarea, .form-group select {
            width: 100%;
            padding: 12px;
            background: #0f1729;
            border: 2px solid #252d4d;
            border-radius: 6px;
            color: #e0e6ed;
            font-size: 1rem;
            font-family: 'Courier New', monospace;
        }
        .form-group input:focus, .form-group textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .form-group textarea {
            min-height: 400px;
            font-size: 0.9rem;
        }
        button {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 6px;
            color: white;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s;
            margin-right: 10px;
        }
        button:hover {
            transform: translateY(-2px);
        }
        button.secondary {
            background: #252d4d;
        }
        .validation-box {
            margin-top: 20px;
            padding: 15px;
            border-radius: 6px;
            display: none;
        }
        .validation-box.error {
            background: #3d1f1f;
            border-left: 4px solid #ef4444;
            display: block;
        }
        .validation-box.success {
            background: #1f3d2f;
            border-left: 4px solid #10b981;
            display: block;
        }
        .validation-box.warning {
            background: #3d301f;
            border-left: 4px solid #f59e0b;
            display: block;
        }
        .message-list {
            list-style: none;
            margin-top: 10px;
        }
        .message-list li {
            padding: 5px 0;
        }
        .message-list li:before {
            content: "• ";
            color: #667eea;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚙️ Configuration Editor</h1>
        
        <div class="sidebar">
            <h3 style="margin-bottom: 15px; color: #8b95a8;">CONFIGURATIONS</h3>
            
            <h4 style="margin: 15px 0 10px; color: #667eea;">Data Generation</h4>
            <div class="config-list" id="data-configs"></div>
            
            <h4 style="margin: 15px 0 10px; color: #667eea;">Training</h4>
            <div class="config-list" id="training-configs"></div>
        </div>
        
        <div class="main-content">
            <h2 style="margin-bottom: 20px;">Edit Configuration</h2>
            
            <div class="form-group">
                <label>Configuration Name</label>
                <input type="text" id="config-name" placeholder="production" />
            </div>
            
            <div class="form-group">
                <label>Category</label>
                <select id="config-category">
                    <option value="data_generation">Data Generation</option>
                    <option value="training">Training</option>
                </select>
            </div>
            
            <div class="form-group">
                <label>Configuration (JSON)</label>
                <textarea id="config-editor" placeholder='{\n  "key": "value"\n}'></textarea>
            </div>
            
            <div>
                <button onclick="validateConfig()">Validate</button>
                <button onclick="saveConfig()">Save</button>
                <button class="secondary" onclick="loadConfig()">Reload</button>
            </div>
            
            <div class="validation-box" id="validation-box"></div>
        </div>
    </div>
    
    <script>
        let currentConfig = '';
        let currentCategory = 'data_generation';
        
        function loadConfigList() {
            fetch('/api/configs')
                .then(r => r.json())
                .then(data => {
                    // Load data generation configs
                    const dataList = document.getElementById('data-configs');
                    dataList.innerHTML = data.data_generation.map(name =>
                        `<div class="config-item" onclick="selectConfig('data_generation', '${name}')">${name}</div>`
                    ).join('');
                    
                    // Load training configs
                    const trainList = document.getElementById('training-configs');
                    trainList.innerHTML = data.training.map(name =>
                        `<div class="config-item" onclick="selectConfig('training', '${name}')">${name}</div>`
                    ).join('');
                });
        }
        
        function selectConfig(category, name) {
            currentConfig = name;
            currentCategory = category;
            
            document.getElementById('config-name').value = name;
            document.getElementById('config-category').value = category;
            
            fetch(`/api/config/${category}/${name}`)
                .then(r => r.json())
                .then(config => {
                    document.getElementById('config-editor').value = JSON.stringify(config, null, 2);
                });
            
            // Update active state
            document.querySelectorAll('.config-item').forEach(el => el.classList.remove('active'));
            event.target.classList.add('active');
        }
        
        function validateConfig() {
            const editor = document.getElementById('config-editor');
            const category = document.getElementById('config-category').value;
            
            try {
                const config = JSON.parse(editor.value);
                
                fetch('/api/validate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({config, category})
                })
                .then(r => r.json())
                .then(result => {
                    const box = document.getElementById('validation-box');
                    
                    if (result.valid) {
                        box.className = 'validation-box success';
                        box.innerHTML = '<strong>✓ Configuration is valid!</strong>';
                    } else {
                        box.className = 'validation-box error';
                        box.innerHTML = '<strong>✗ Validation Issues:</strong><ul class="message-list">' +
                            result.issues.map(i => `<li>${i}</li>`).join('') +
                            '</ul>';
                    }
                    
                    if (result.recommendations && result.recommendations.length > 0) {
                        box.className = 'validation-box warning';
                        box.innerHTML += '<strong style="margin-top:15px;display:block;">💡 Recommendations:</strong><ul class="message-list">' +
                            result.recommendations.map(r => `<li>${r}</li>`).join('') +
                            '</ul>';
                    }
                });
                
            } catch (e) {
                const box = document.getElementById('validation-box');
                box.className = 'validation-box error';
                box.innerHTML = `<strong>✗ JSON Parse Error:</strong><p style="margin-top:10px;">${e.message}</p>`;
            }
        }
        
        function saveConfig() {
            const name = document.getElementById('config-name').value;
            const category = document.getElementById('config-category').value;
            const editor = document.getElementById('config-editor');
            
            try {
                const config = JSON.parse(editor.value);
                
                fetch('/api/save', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name, category, config})
                })
                .then(r => r.json())
                .then(result => {
                    const box = document.getElementById('validation-box');
                    box.className = 'validation-box success';
                    box.innerHTML = `<strong>✓ ${result.message}</strong>`;
                    
                    loadConfigList();
                });
                
            } catch (e) {
                alert('Invalid JSON: ' + e.message);
            }
        }
        
        function loadConfig() {
            selectConfig(currentCategory, currentConfig);
        }
        
        // Initialize
        loadConfigList();
    </script>
</body>
</html>
            """
        
        try:
            with socketserver.TCPServer(("", self.port), EditorHandler) as httpd:
                logger.info(f"Configuration editor started at http://localhost:{self.port}")
                logger.info("Press Ctrl+C to stop")
                httpd.serve_forever()
        except Exception as e:
            logger.error(f"Failed to start server: {e}")


def main():
    parser = argparse.ArgumentParser(description='Web-based configuration editor')
    parser.add_argument('--port', type=int, default=8090,
                       help='Server port (default: 8090)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("WEB-BASED CONFIGURATION EDITOR")
    print("="*70)
    print()
    print(f"Starting server on port {args.port}...")
    print()
    
    editor = ConfigEditorServer(args.port)
    editor.start()


if __name__ == '__main__':
    main()
