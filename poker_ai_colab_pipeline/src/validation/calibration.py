"""
Calibration Testing for Value Network

Tests Expected Calibration Error (ECE) and applies temperature scaling
to ensure well-calibrated probability estimates.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np


class CalibrationTester:
    """
    Test and calibrate value network predictions.
    
    Implements:
    - Expected Calibration Error (ECE) computation
    - Temperature scaling for calibration
    - Per-bin accuracy analysis
    
    Usage:
        tester = CalibrationTester()
        ece = tester.compute_ece(predictions, targets)
        optimal_temp = tester.find_optimal_temperature(predictions, targets)
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize calibration tester.
        
        Args:
            n_bins: Number of bins for ECE computation
        """
        self.n_bins = n_bins
    
    def compute_ece(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        confidences: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        For regression (value estimation), we use a simplified ECE
        based on prediction error magnitudes.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            confidences: Optional confidence scores
            
        Returns:
            ECE value (lower is better, 0 is perfectly calibrated)
        """
        predictions = np.asarray(predictions).flatten()
        targets = np.asarray(targets).flatten()
        
        if confidences is None:
            # Use prediction magnitudes as proxy for confidence
            confidences = np.abs(predictions)
            confidences = confidences / (np.max(confidences) + 1e-10)
        
        # Bin by confidence
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        
        for i in range(self.n_bins):
            in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            n_in_bin = np.sum(in_bin)
            
            if n_in_bin == 0:
                continue
            
            # For regression: accuracy is inverse of normalized error
            errors = np.abs(predictions[in_bin] - targets[in_bin])
            avg_error = np.mean(errors)
            avg_conf = np.mean(confidences[in_bin])
            
            # Normalized error (scale by target range)
            target_range = np.max(np.abs(targets)) + 1e-10
            normalized_error = avg_error / target_range
            accuracy = 1 - normalized_error
            
            ece += (n_in_bin / len(predictions)) * np.abs(accuracy - avg_conf)
        
        return ece
    
    def find_optimal_temperature(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        search_range: Tuple[float, float] = (0.5, 2.0),
        n_steps: int = 31
    ) -> float:
        """
        Find optimal temperature for calibration.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            search_range: Range of temperatures to search
            n_steps: Number of steps in grid search
            
        Returns:
            Optimal temperature value
        """
        best_temp = 1.0
        best_ece = float('inf')
        
        for temp in np.linspace(search_range[0], search_range[1], n_steps):
            scaled = predictions / temp
            ece = self.compute_ece(scaled, targets)
            
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
        
        return best_temp
    
    def get_calibration_curve(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict:
        """
        Get calibration curve data for plotting.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            
        Returns:
            Dictionary with bin data for plotting
        """
        predictions = np.asarray(predictions).flatten()
        targets = np.asarray(targets).flatten()
        
        # Use prediction magnitudes as confidence proxy
        confidences = np.abs(predictions)
        confidences = confidences / (np.max(confidences) + 1e-10)
        
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        accuracies = []
        avg_confidences = []
        bin_sizes = []
        
        for i in range(self.n_bins):
            in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            n_in_bin = np.sum(in_bin)
            bin_sizes.append(n_in_bin)
            
            if n_in_bin == 0:
                accuracies.append(np.nan)
                avg_confidences.append(np.nan)
                continue
            
            errors = np.abs(predictions[in_bin] - targets[in_bin])
            target_range = np.max(np.abs(targets)) + 1e-10
            accuracy = 1 - np.mean(errors) / target_range
            
            accuracies.append(accuracy)
            avg_confidences.append(np.mean(confidences[in_bin]))
        
        return {
            'bin_centers': bin_centers,
            'accuracies': np.array(accuracies),
            'confidences': np.array(avg_confidences),
            'bin_sizes': np.array(bin_sizes)
        }
    
    def run_calibration_test(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        threshold: float = 0.05
    ) -> Dict:
        """
        Run full calibration test.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            threshold: ECE threshold for passing
            
        Returns:
            Test results dictionary
        """
        ece = self.compute_ece(predictions, targets)
        optimal_temp = self.find_optimal_temperature(predictions, targets)
        
        # Compute calibrated ECE
        calibrated_predictions = predictions / optimal_temp
        calibrated_ece = self.compute_ece(calibrated_predictions, targets)
        
        return {
            'name': 'Value Network Calibration',
            'passed': calibrated_ece < threshold,
            'ece_before': ece,
            'ece_after': calibrated_ece,
            'optimal_temperature': optimal_temp,
            'threshold': threshold,
            'improvement': (ece - calibrated_ece) / (ece + 1e-10)
        }


__all__ = ['CalibrationTester']
