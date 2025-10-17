#!/usr/bin/env python3
"""
Production Workflow Orchestrator

Automated end-to-end workflow management:
- Orchestrates data generation, training, and validation
- Real-time progress tracking
- Automated hyperparameter tuning
- Multi-GPU resource allocation
- Intelligent error recovery
- Performance monitoring and recommendations

Usage:
  # Full automated workflow
  python scripts/orchestrator.py --workflow full --profile championship
  
  # Data generation only
  python scripts/orchestrator.py --workflow generation --profile production
  
  # Training with auto-tuning
  python scripts/orchestrator.py --workflow training --auto-tune
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
import subprocess
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """Intelligent workflow orchestration with automation and monitoring."""
    
    def __init__(self, profile: str, config: Optional[Dict] = None):
        self.profile = profile
        self.config = config or {}
        self.workflow_state = {
            'start_time': time.time(),
            'stages_completed': [],
            'current_stage': None,
            'errors': [],
            'recommendations': []
        }
        self.results = {}
        
    def execute_workflow(self, workflow_type: str, **kwargs):
        """Execute complete workflow with monitoring and error handling."""
        logger.info(f"Starting {workflow_type} workflow with profile: {self.profile}")
        
        workflows = {
            'full': self._workflow_full,
            'generation': self._workflow_generation,
            'training': self._workflow_training,
            'validation': self._workflow_validation
        }
        
        workflow_func = workflows.get(workflow_type)
        if not workflow_func:
            raise ValueError(f"Unknown workflow: {workflow_type}")
        
        try:
            # Execute workflow
            result = workflow_func(**kwargs)
            
            # Final analysis
            self._analyze_results()
            
            # Generate report
            self._generate_report()
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            self.workflow_state['errors'].append({
                'stage': self.workflow_state['current_stage'],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    def _workflow_full(self, auto_tune: bool = False, multi_gpu: bool = False):
        """Complete end-to-end workflow."""
        logger.info("Executing full workflow: generation → training → validation")
        
        # Stage 1: Data Generation
        self._execute_stage('generation', self._stage_generation)
        
        # Stage 2: Optional hyperparameter tuning
        if auto_tune:
            self._execute_stage('auto_tune', self._stage_auto_tune)
        
        # Stage 3: Training
        self._execute_stage('training', lambda: self._stage_training(multi_gpu=multi_gpu))
        
        # Stage 4: Validation
        self._execute_stage('validation', self._stage_validation)
        
        # Stage 5: Analysis
        self._execute_stage('analysis', self._stage_analysis)
        
        return self.results
    
    def _workflow_generation(self, **kwargs):
        """Data generation workflow."""
        self._execute_stage('generation', self._stage_generation)
        self._execute_stage('validation', lambda: self._stage_validation(data_only=True))
        return self.results
    
    def _workflow_training(self, auto_tune: bool = False, multi_gpu: bool = False):
        """Training workflow."""
        if auto_tune:
            self._execute_stage('auto_tune', self._stage_auto_tune)
        
        self._execute_stage('training', lambda: self._stage_training(multi_gpu=multi_gpu))
        self._execute_stage('validation', self._stage_validation)
        self._execute_stage('analysis', self._stage_analysis)
        return self.results
    
    def _workflow_validation(self):
        """Validation workflow."""
        self._execute_stage('validation', self._stage_validation)
        self._execute_stage('analysis', self._stage_analysis)
        return self.results
    
    def _execute_stage(self, stage_name: str, stage_func):
        """Execute a workflow stage with error handling."""
        self.workflow_state['current_stage'] = stage_name
        logger.info(f"Executing stage: {stage_name}")
        
        try:
            start_time = time.time()
            result = stage_func()
            elapsed = time.time() - start_time
            
            self.workflow_state['stages_completed'].append({
                'stage': stage_name,
                'duration': elapsed,
                'timestamp': datetime.now().isoformat()
            })
            
            self.results[stage_name] = result
            logger.info(f"Stage '{stage_name}' completed in {elapsed:.1f}s")
            
        except Exception as e:
            logger.error(f"Stage '{stage_name}' failed: {e}")
            self.workflow_state['errors'].append({
                'stage': stage_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            raise
    
    def _stage_generation(self) -> Dict:
        """Execute data generation stage."""
        cmd = [
            sys.executable,
            'scripts/generate_data.py',
            '--profile', self.profile,
            '--yes'
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Data generation failed: {result.stderr}")
        
        return {'status': 'success', 'output': result.stdout}
    
    def _stage_auto_tune(self) -> Dict:
        """Execute hyperparameter tuning stage."""
        cmd = [
            sys.executable,
            'scripts/auto_tune.py',
            '--task', 'training',
            '--trials', '30',
            '--output', 'optimization_results.json'
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"Auto-tuning completed with warnings: {result.stderr}")
        
        # Load optimized parameters
        opt_file = Path('optimization_results.json')
        if opt_file.exists():
            with open(opt_file, 'r') as f:
                opt_results = json.load(f)
            return opt_results
        
        return {'status': 'completed'}
    
    def _stage_training(self, multi_gpu: bool = False) -> Dict:
        """Execute training stage."""
        if multi_gpu:
            # Check GPU availability
            try:
                import torch
                gpu_count = torch.cuda.device_count()
                
                if gpu_count > 1:
                    cmd = [
                        'torchrun',
                        f'--nproc_per_node={gpu_count}',
                        'scripts/train_model.py',
                        '--profile', self.profile,
                        '--use-gpu'
                    ]
                else:
                    logger.warning(f"Multi-GPU requested but only {gpu_count} GPU available")
                    cmd = [
                        sys.executable,
                        'scripts/train_model.py',
                        '--profile', self.profile,
                        '--use-gpu'
                    ]
            except ImportError:
                logger.warning("PyTorch not available, using CPU")
                cmd = [
                    sys.executable,
                    'scripts/train_model.py',
                    '--profile', self.profile
                ]
        else:
            cmd = [
                sys.executable,
                'scripts/train_model.py',
                '--profile', self.profile,
                '--use-gpu'
            ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Training failed: {result.stderr}")
        
        return {'status': 'success', 'output': result.stdout}
    
    def _stage_validation(self, data_only: bool = False) -> Dict:
        """Execute validation stage."""
        validation_type = 'data' if data_only else 'all'
        
        cmd = [
            sys.executable,
            'scripts/validate_model.py',
            '--type', validation_type
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"Validation completed with issues: {result.stderr}")
        
        return {'status': 'completed', 'output': result.stdout}
    
    def _stage_analysis(self) -> Dict:
        """Execute AI analysis stage."""
        cmd = [
            sys.executable,
            'scripts/ai_analyzer.py',
            '--type', 'full',
            '--output', 'workflow_analysis.json'
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Load analysis results
        analysis_file = Path('workflow_analysis.json')
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            return analysis
        
        return {'status': 'completed'}
    
    def _analyze_results(self):
        """Analyze workflow results and generate recommendations."""
        logger.info("Analyzing workflow results...")
        
        # Check for failures
        if self.workflow_state['errors']:
            self.workflow_state['recommendations'].append({
                'priority': 'critical',
                'message': f"{len(self.workflow_state['errors'])} stage(s) failed",
                'action': 'Review error logs and retry failed stages'
            })
        
        # Check performance
        total_time = time.time() - self.workflow_state['start_time']
        if total_time > 86400:  # >1 day
            self.workflow_state['recommendations'].append({
                'priority': 'high',
                'message': f'Workflow took {total_time/3600:.1f} hours',
                'action': 'Consider using multi-GPU training or distributed generation'
            })
        
        # Check results quality
        if 'analysis' in self.results:
            analysis = self.results['analysis']
            if 'summary' in analysis:
                if analysis['summary'].get('overall_status') == 'poor':
                    self.workflow_state['recommendations'].append({
                        'priority': 'critical',
                        'message': 'Poor overall performance detected',
                        'action': 'Review analysis recommendations and regenerate data/retrain'
                    })
    
    def _generate_report(self):
        """Generate comprehensive workflow report."""
        report = {
            'workflow_type': 'automated',
            'profile': self.profile,
            'start_time': datetime.fromtimestamp(self.workflow_state['start_time']).isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_duration': time.time() - self.workflow_state['start_time'],
            'stages_completed': self.workflow_state['stages_completed'],
            'errors': self.workflow_state['errors'],
            'recommendations': self.workflow_state['recommendations'],
            'results': {k: v for k, v in self.results.items() if k != 'output'}
        }
        
        report_file = Path('workflow_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Workflow report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*70)
        print("WORKFLOW EXECUTION REPORT")
        print("="*70)
        print(f"\nProfile: {self.profile}")
        print(f"Total Duration: {report['total_duration']/3600:.2f} hours")
        print(f"Stages Completed: {len(self.workflow_state['stages_completed'])}")
        print(f"Errors: {len(self.workflow_state['errors'])}")
        
        if self.workflow_state['recommendations']:
            print("\nRecommendations:")
            for rec in self.workflow_state['recommendations']:
                print(f"  [{rec['priority'].upper()}] {rec['message']}")
                print(f"    → {rec['action']}")
        
        print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Automated workflow orchestration',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--workflow', type=str, required=True,
                       choices=['full', 'generation', 'training', 'validation'],
                       help='Workflow to execute')
    parser.add_argument('--profile', type=str, default='production',
                       choices=['testing', 'development', 'production', 'championship'],
                       help='Configuration profile')
    parser.add_argument('--auto-tune', action='store_true',
                       help='Enable automatic hyperparameter tuning')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Enable multi-GPU training')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PRODUCTION WORKFLOW ORCHESTRATOR")
    print("="*70)
    print()
    print(f"Workflow: {args.workflow}")
    print(f"Profile: {args.profile}")
    print(f"Auto-tune: {args.auto_tune}")
    print(f"Multi-GPU: {args.multi_gpu}")
    print()
    print("Starting automated workflow...")
    print("="*70)
    print()
    
    # Create orchestrator
    orchestrator = WorkflowOrchestrator(args.profile)
    
    # Execute workflow
    try:
        result = orchestrator.execute_workflow(
            args.workflow,
            auto_tune=args.auto_tune,
            multi_gpu=args.multi_gpu
        )
        
        print()
        print("✓ Workflow completed successfully!")
        
    except Exception as e:
        print()
        print(f"✗ Workflow failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
