#!/usr/bin/env python3
"""
Automated Analysis and Recommendation Engine

Advanced AI-driven system for analyzing training/generation workflows and providing:
- Deep performance analysis
- Bottleneck identification
- Optimization recommendations
- Predictive modeling
- Comparative benchmarking

Usage:
  # Analyze training run
  python scripts/ai_analyzer.py --type training --data models/checkpoints
  
  # Analyze data generation
  python scripts/ai_analyzer.py --type generation --data src/train_samples
  
  # Comprehensive analysis
  python scripts/ai_analyzer.py --type full --output analysis_report.json
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
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIAnalyzer:
    """AI-driven analyzer for training workflows with deep insights."""
    
    def __init__(self):
        self.analyses = []
        self.recommendations = []
        self.benchmarks = self._load_benchmarks()
    
    def _load_benchmarks(self) -> Dict:
        """Load performance benchmarks for comparison."""
        return {
            'training': {
                'correlation': {
                    'championship': 0.85,
                    'production': 0.75,
                    'development': 0.65,
                    'minimum': 0.50
                },
                'samples_per_sec': {
                    'gpu_high': 50.0,
                    'gpu_medium': 20.0,
                    'gpu_low': 10.0,
                    'cpu': 2.0
                },
                'convergence_epochs': {
                    'fast': 50,
                    'normal': 100,
                    'slow': 200
                }
            },
            'generation': {
                'samples_per_sec': {
                    'high': 3.0,
                    'medium': 2.0,
                    'low': 1.0
                },
                'cfr_quality': {
                    'championship': 2500,
                    'production': 2000,
                    'minimum': 1500
                }
            }
        }
    
    def analyze_training(self, data_path: str) -> Dict:
        """Comprehensive training analysis."""
        logger.info("Analyzing training performance...")
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'type': 'training',
            'metrics': {},
            'insights': [],
            'recommendations': [],
            'score': 0.0
        }
        
        try:
            # Load training metrics if available
            metrics_file = Path(data_path) / 'training_metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                analysis['metrics'] = metrics
                
                # Analyze correlation
                if 'final_correlation' in metrics:
                    corr = metrics['final_correlation']
                    analysis['metrics']['correlation'] = corr
                    
                    # Compare to benchmarks
                    if corr >= self.benchmarks['training']['correlation']['championship']:
                        analysis['insights'].append({
                            'category': 'performance',
                            'level': 'excellent',
                            'message': f"Championship-level correlation: {corr:.3f}",
                            'detail': "Model achieves top-tier performance"
                        })
                        analysis['score'] += 0.4
                    elif corr >= self.benchmarks['training']['correlation']['production']:
                        analysis['insights'].append({
                            'category': 'performance',
                            'level': 'good',
                            'message': f"Production-level correlation: {corr:.3f}",
                            'detail': "Suitable for real-world deployment"
                        })
                        analysis['score'] += 0.3
                    else:
                        analysis['insights'].append({
                            'category': 'performance',
                            'level': 'poor',
                            'message': f"Low correlation: {corr:.3f}",
                            'detail': "Needs improvement for production use"
                        })
                        analysis['recommendations'].append({
                            'priority': 'critical',
                            'category': 'data_quality',
                            'action': 'Generate more training data',
                            'rationale': 'Low correlation indicates insufficient data or quality issues',
                            'steps': [
                                'Generate 100K+ samples with CFR iterations >= 2000',
                                'Enable championship bet sizing',
                                'Verify data quality with validate_model.py'
                            ]
                        })
                
                # Analyze training speed
                if 'training_time' in metrics and 'epochs' in metrics:
                    time_per_epoch = metrics['training_time'] / metrics['epochs']
                    
                    if time_per_epoch > 300:  # >5 minutes per epoch
                        analysis['recommendations'].append({
                            'priority': 'high',
                            'category': 'efficiency',
                            'action': 'Optimize training speed',
                            'rationale': f'Training is slow ({time_per_epoch:.1f}s per epoch)',
                            'steps': [
                                'Enable GPU training with --use-gpu',
                                'Use larger batch sizes if memory allows',
                                'Consider multi-GPU training for large datasets'
                            ]
                        })
                
                # Analyze convergence
                if 'loss_history' in metrics:
                    loss_history = metrics['loss_history']
                    if len(loss_history) > 20:
                        # Check for plateau
                        recent_loss = np.mean(loss_history[-10:])
                        older_loss = np.mean(loss_history[-20:-10])
                        
                        if abs(recent_loss - older_loss) / older_loss < 0.01:
                            analysis['insights'].append({
                                'category': 'convergence',
                                'level': 'warning',
                                'message': 'Training has plateaued',
                                'detail': 'Loss not improving in recent epochs'
                            })
                            analysis['recommendations'].append({
                                'priority': 'medium',
                                'category': 'hyperparameters',
                                'action': 'Adjust learning rate schedule',
                                'rationale': 'Model may have converged or hit local minimum',
                                'steps': [
                                    'Reduce learning rate by 50%',
                                    'Enable cosine annealing',
                                    'Consider early stopping'
                                ]
                            })
                
        except Exception as e:
            logger.error(f"Error analyzing training: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def analyze_data_generation(self, data_path: str) -> Dict:
        """Comprehensive data generation analysis."""
        logger.info("Analyzing data generation...")
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'type': 'generation',
            'metrics': {},
            'insights': [],
            'recommendations': [],
            'score': 0.0
        }
        
        try:
            import torch
            
            data_path = Path(data_path)
            
            # Check data files
            required_files = [
                'train_inputs.pt', 'train_targets.pt', 'train_mask.pt',
                'valid_inputs.pt', 'valid_targets.pt'
            ]
            
            missing_files = [f for f in required_files if not (data_path / f).exists()]
            
            if missing_files:
                analysis['insights'].append({
                    'category': 'data_quality',
                    'level': 'error',
                    'message': f'Missing files: {", ".join(missing_files)}',
                    'detail': 'Data generation incomplete or failed'
                })
                return analysis
            
            # Load and analyze data
            train_inputs = torch.load(data_path / 'train_inputs.pt')
            train_targets = torch.load(data_path / 'train_targets.pt')
            train_mask = torch.load(data_path / 'train_mask.pt')
            
            # Sample count analysis
            n_samples = len(train_inputs)
            analysis['metrics']['sample_count'] = n_samples
            
            if n_samples >= 500000:
                analysis['insights'].append({
                    'category': 'data_quantity',
                    'level': 'excellent',
                    'message': f'Championship-level sample count: {n_samples:,}',
                    'detail': 'Sufficient data for top-tier performance'
                })
                analysis['score'] += 0.4
            elif n_samples >= 100000:
                analysis['insights'].append({
                    'category': 'data_quantity',
                    'level': 'good',
                    'message': f'Production-level sample count: {n_samples:,}',
                    'detail': 'Good data quantity for real-world use'
                })
                analysis['score'] += 0.3
            elif n_samples >= 50000:
                analysis['insights'].append({
                    'category': 'data_quantity',
                    'level': 'acceptable',
                    'message': f'Acceptable sample count: {n_samples:,}',
                    'detail': 'Minimum for development'
                })
                analysis['score'] += 0.2
            else:
                analysis['insights'].append({
                    'category': 'data_quantity',
                    'level': 'poor',
                    'message': f'Insufficient samples: {n_samples:,}',
                    'detail': 'Too few samples for reliable training'
                })
                analysis['recommendations'].append({
                    'priority': 'critical',
                    'category': 'data_generation',
                    'action': 'Generate more samples',
                    'rationale': f'Current {n_samples:,} samples is below minimum threshold',
                    'steps': [
                        'Target: 100,000+ samples for production',
                        'Use: python scripts/generate_data.py --profile production',
                        'Or: python scripts/generate_data.py --profile championship for best results'
                    ]
                })
            
            # Data quality analysis
            mask_ratio = train_mask.float().mean().item()
            analysis['metrics']['mask_ratio'] = mask_ratio
            
            if mask_ratio < 0.5:
                analysis['insights'].append({
                    'category': 'data_quality',
                    'level': 'warning',
                    'message': f'High masking ratio: {(1-mask_ratio)*100:.1f}% masked',
                    'detail': 'Many invalid actions, may indicate generation issues'
                })
            
            # Street distribution analysis
            if (data_path / 'train_street.pt').exists():
                train_street = torch.load(data_path / 'train_street.pt')
                street_dist = {}
                for s in [0, 1, 2, 3]:
                    count = (train_street == s).sum().item()
                    pct = 100.0 * count / len(train_street)
                    street_dist[['preflop', 'flop', 'turn', 'river'][s]] = {
                        'count': count,
                        'percentage': pct
                    }
                
                analysis['metrics']['street_distribution'] = street_dist
                
                # Check for postflop emphasis
                postflop_pct = sum(street_dist[s]['percentage'] for s in ['flop', 'turn', 'river'])
                if postflop_pct >= 60:
                    analysis['insights'].append({
                        'category': 'data_quality',
                        'level': 'good',
                        'message': f'Good postflop emphasis: {postflop_pct:.1f}%',
                        'detail': 'Matches real poker play distribution'
                    })
                    analysis['score'] += 0.2
        
        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def generate_report(self, analyses: List[Dict]) -> Dict:
        """Generate comprehensive report with all analyses."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'analyses': analyses,
            'summary': {
                'total_insights': sum(len(a.get('insights', [])) for a in analyses),
                'total_recommendations': sum(len(a.get('recommendations', [])) for a in analyses),
                'average_score': np.mean([a.get('score', 0) for a in analyses])
            },
            'priority_actions': []
        }
        
        # Collect all recommendations and prioritize
        all_recommendations = []
        for analysis in analyses:
            for rec in analysis.get('recommendations', []):
                rec['source'] = analysis['type']
                all_recommendations.append(rec)
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_recommendations.sort(key=lambda x: priority_order.get(x.get('priority', 'low'), 999))
        
        report['priority_actions'] = all_recommendations[:10]  # Top 10
        
        # Overall assessment
        avg_score = report['summary']['average_score']
        if avg_score >= 0.8:
            report['summary']['overall_status'] = 'excellent'
            report['summary']['message'] = 'System is performing at championship level'
        elif avg_score >= 0.6:
            report['summary']['overall_status'] = 'good'
            report['summary']['message'] = 'System is production-ready with room for optimization'
        elif avg_score >= 0.4:
            report['summary']['overall_status'] = 'acceptable'
            report['summary']['message'] = 'System functional but needs improvements'
        else:
            report['summary']['overall_status'] = 'poor'
            report['summary']['message'] = 'Critical issues detected - immediate action required'
        
        return report
    
    def save_report(self, report: Dict, filepath: str):
        """Save analysis report."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='AI-driven analysis and recommendations')
    parser.add_argument('--type', type=str, required=True,
                       choices=['training', 'generation', 'full'],
                       help='Analysis type')
    parser.add_argument('--data', type=str, help='Data directory to analyze')
    parser.add_argument('--output', type=str, default='analysis_report.json',
                       help='Output report file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("AI-DRIVEN ANALYSIS ENGINE")
    print("="*70)
    print()
    print(f"Analysis type: {args.type}")
    if args.data:
        print(f"Data path: {args.data}")
    print()
    print("Running analysis...")
    print()
    
    analyzer = AIAnalyzer()
    analyses = []
    
    if args.type in ['training', 'full']:
        train_analysis = analyzer.analyze_training(args.data or 'models/checkpoints')
        analyses.append(train_analysis)
    
    if args.type in ['generation', 'full']:
        data_analysis = analyzer.analyze_data_generation(args.data or 'src/train_samples')
        analyses.append(data_analysis)
    
    # Generate comprehensive report
    report = analyzer.generate_report(analyses)
    
    # Save report
    analyzer.save_report(report, args.output)
    
    # Print summary
    print("="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print()
    print(f"Overall Status: {report['summary']['overall_status'].upper()}")
    print(f"Message: {report['summary']['message']}")
    print()
    print(f"Total Insights: {report['summary']['total_insights']}")
    print(f"Total Recommendations: {report['summary']['total_recommendations']}")
    print()
    
    if report['priority_actions']:
        print("Top Priority Actions:")
        for i, action in enumerate(report['priority_actions'][:5], 1):
            print(f"  {i}. [{action['priority'].upper()}] {action['action']}")
            print(f"     {action['rationale']}")
        print()
    
    print(f"Full report saved to: {args.output}")
    print("="*70)


if __name__ == '__main__':
    main()
