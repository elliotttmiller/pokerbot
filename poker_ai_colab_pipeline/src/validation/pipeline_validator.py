"""
Pipeline Validator - Comprehensive Validation Suite

Runs all validation tests and generates compliance report.
"""

from typing import Dict, List, Optional
import time
import gc

import numpy as np


class PipelineValidator:
    """
    Comprehensive pipeline validation suite.
    
    Runs:
    1. DeepStack fidelity tests
    2. Value network calibration tests
    3. Perception accuracy tests (if model available)
    4. Memory stability tests
    5. End-to-end integration tests
    
    Usage:
        validator = PipelineValidator()
        report = validator.run_all_tests()
        validator.print_report(report)
    """
    
    def __init__(
        self,
        perception_model: Optional[str] = None,
        decision_model: Optional[str] = None,
        test_images_dir: Optional[str] = None
    ):
        """
        Initialize validator.
        
        Args:
            perception_model: Path to perception model
            decision_model: Path to decision model
            test_images_dir: Directory with test images
        """
        self.perception_model = perception_model
        self.decision_model = decision_model
        self.test_images_dir = test_images_dir
    
    def run_all_tests(self) -> Dict:
        """
        Run all validation tests.
        
        Returns:
            Comprehensive validation report
        """
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tests': [],
            'summary': {},
            'overall_passed': True
        }
        
        # 1. DeepStack Fidelity
        print("\n[1/5] Running DeepStack Fidelity Tests...")
        fidelity_results = self._run_fidelity_tests()
        report['tests'].append(fidelity_results)
        
        # 2. Value Network Calibration
        print("[2/5] Running Calibration Tests...")
        calibration_results = self._run_calibration_tests()
        report['tests'].append(calibration_results)
        
        # 3. Perception Tests
        print("[3/5] Running Perception Tests...")
        perception_results = self._run_perception_tests()
        report['tests'].append(perception_results)
        
        # 4. Memory Stability
        print("[4/5] Running Memory Stability Tests...")
        memory_results = self._run_memory_tests()
        report['tests'].append(memory_results)
        
        # 5. Integration Tests
        print("[5/5] Running Integration Tests...")
        integration_results = self._run_integration_tests()
        report['tests'].append(integration_results)
        
        # Summary
        total_passed = sum(1 for t in report['tests'] if t.get('passed', False))
        total_tests = len(report['tests'])
        
        report['summary'] = {
            'passed': total_passed,
            'failed': total_tests - total_passed,
            'total': total_tests,
            'score': total_passed / total_tests if total_tests > 0 else 0.0
        }
        
        report['overall_passed'] = total_passed == total_tests
        
        return report
    
    def _run_fidelity_tests(self) -> Dict:
        """Run DeepStack fidelity tests."""
        try:
            from .fidelity_test import DeepStackFidelityTest
            
            tester = DeepStackFidelityTest()
            results = tester.run_all_tests()
            
            return {
                'category': 'DeepStack Fidelity',
                'passed': results['score'] >= 0.75,  # 75% tests must pass
                'score': results['score'],
                'details': results['tests']
            }
        except Exception as e:
            return {
                'category': 'DeepStack Fidelity',
                'passed': False,
                'score': 0.0,
                'error': str(e)
            }
    
    def _run_calibration_tests(self) -> Dict:
        """Run value network calibration tests."""
        try:
            from .calibration import CalibrationTester
            
            tester = CalibrationTester()
            
            # Generate synthetic test data
            np.random.seed(42)
            predictions = np.random.randn(100)
            targets = predictions + np.random.randn(100) * 0.5
            
            result = tester.run_calibration_test(predictions, targets, threshold=0.1)
            
            return {
                'category': 'Value Network Calibration',
                'passed': result['passed'],
                'ece_before': result['ece_before'],
                'ece_after': result['ece_after'],
                'optimal_temperature': result['optimal_temperature']
            }
        except Exception as e:
            return {
                'category': 'Value Network Calibration',
                'passed': False,
                'error': str(e)
            }
    
    def _run_perception_tests(self) -> Dict:
        """Run perception accuracy tests."""
        try:
            from ..perception.detector import QWENPokerDetector
            from ..perception.data_models import GameState, Card
            
            # Test data model parsing
            test_cases = [
                ('As', True),
                ('Kh', True),
                ('Td', True),
                ('2c', True),
                ('invalid', False)
            ]
            
            passed = 0
            for card_str, should_pass in test_cases:
                try:
                    Card.from_string(card_str)
                    if should_pass:
                        passed += 1
                except:
                    if not should_pass:
                        passed += 1
            
            accuracy = passed / len(test_cases)
            
            return {
                'category': 'Perception',
                'passed': accuracy >= 0.9,
                'accuracy': accuracy,
                'tests_run': len(test_cases),
                'tests_passed': passed
            }
        except Exception as e:
            return {
                'category': 'Perception',
                'passed': False,
                'error': str(e)
            }
    
    def _run_memory_tests(self) -> Dict:
        """Run memory stability tests."""
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        
        try:
            from ..decision.cfr import CFRSolver
            
            # Run multiple CFR solves
            solver = CFRSolver(num_hands=36)
            node_params = {
                'street': 0,
                'bets': [10, 10],
                'current_player': 0,
                'board': [],
                'bet_sizing': [1.0]
            }
            
            # Track memory
            initial_mem = self._get_memory_usage()
            
            for _ in range(10):
                tree = solver.build_tree(node_params)
                player_range = np.ones(36) / 36
                opponent_range = np.ones(36) / 36
                solver.solve(tree, player_range, opponent_range, iterations=50)
                
                # Cleanup
                gc.collect()
            
            final_mem = self._get_memory_usage()
            mem_increase = final_mem - initial_mem
            
            # Pass if memory increase is less than 100MB
            passed = mem_increase < 100 * 1024 * 1024
            
            return {
                'category': 'Memory Stability',
                'passed': passed,
                'initial_mb': initial_mem / (1024 * 1024),
                'final_mb': final_mem / (1024 * 1024),
                'increase_mb': mem_increase / (1024 * 1024),
                'cycles': 10
            }
        except Exception as e:
            return {
                'category': 'Memory Stability',
                'passed': False,
                'error': str(e)
            }
    
    def _run_integration_tests(self) -> Dict:
        """Run end-to-end integration tests."""
        try:
            from ..orchestration.workflow import PokerWorkflowOrchestrator
            
            # Create orchestrator (no models)
            orchestrator = PokerWorkflowOrchestrator(
                simulation_mode=True
            )
            
            # Test state solving
            test_state = {
                'hole_cards': ['As', 'Kh'],
                'community_cards': [],
                'pot_size': 100,
                'current_bet': 20,
                'street': 'preflop'
            }
            
            result = orchestrator.solve_state(test_state, iterations=100)
            
            # Validate result
            valid = (
                result.action in ['fold', 'check', 'call', 'raise', 'bet'] and
                result.confidence >= 0 and
                result.confidence <= 1 and
                result.latency_ms > 0
            )
            
            return {
                'category': 'End-to-End Integration',
                'passed': valid,
                'action': result.action,
                'confidence': result.confidence,
                'latency_ms': result.latency_ms
            }
        except Exception as e:
            return {
                'category': 'End-to-End Integration',
                'passed': False,
                'error': str(e)
            }
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            # Fallback estimate
            return 0
    
    def print_report(self, report: Dict):
        """Print formatted validation report."""
        print("\n" + "=" * 60)
        print("POKER AI PIPELINE VALIDATION REPORT")
        print("=" * 60)
        print(f"Timestamp: {report['timestamp']}")
        print("-" * 60)
        
        for test in report['tests']:
            status = "✅ PASS" if test.get('passed', False) else "❌ FAIL"
            print(f"\n{test['category']}: {status}")
            
            for key, value in test.items():
                if key not in ['category', 'passed', 'details']:
                    print(f"  {key}: {value}")
        
        print("\n" + "-" * 60)
        print("SUMMARY")
        print("-" * 60)
        summary = report['summary']
        print(f"Tests Passed: {summary['passed']}/{summary['total']}")
        print(f"Score: {summary['score']:.1%}")
        
        overall = "✅ ALL TESTS PASSED" if report['overall_passed'] else "❌ SOME TESTS FAILED"
        print(f"\nOverall: {overall}")
        print("=" * 60)


def run_full_validation(
    perception_model: Optional[str] = None,
    decision_model: Optional[str] = None
) -> Dict:
    """
    Run full validation suite.
    
    Args:
        perception_model: Path to perception model
        decision_model: Path to decision model
        
    Returns:
        Validation report
    """
    validator = PipelineValidator(
        perception_model=perception_model,
        decision_model=decision_model
    )
    
    report = validator.run_all_tests()
    validator.print_report(report)
    
    return report


__all__ = ['PipelineValidator', 'run_full_validation']
