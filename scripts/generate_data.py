#!/usr/bin/env python3
"""
Unified Training Data Generation Script

This script consolidates all data generation functionality with profile-based configurations.
Replaces: generate_quick_data.py, generate_production_data.py

Usage:
  # Using predefined profiles
  python scripts/generate_data.py --profile testing
  python scripts/generate_data.py --profile development
  python scripts/generate_data.py --profile production
  python scripts/generate_data.py --profile championship
  
  # Custom configuration
  python scripts/generate_data.py --config config/data_generation/custom.json
  
  # Override specific parameters
  python scripts/generate_data.py --profile production --samples 200000 --cfr-iters 3000
  
  # With advanced features
  python scripts/generate_data.py --profile championship --adaptive-cfr --bucket-weights weights.json
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
from pathlib import Path
from typing import Dict, Optional
from deepstack.data.data_generation import generate_training_data
import numpy as np


# Predefined profiles for different use cases
PROFILES = {
    'testing': {
        'samples': 1000,
        'validation_samples': 200,
        'cfr_iterations': 500,
        'output': 'src/train_samples_test',
        'championship_bet_sizing': False,
        'adaptive_cfr': False,
        'description': 'Quick testing (5-10 minutes)',
    },
    'development': {
        'samples': 10000,
        'validation_samples': 2000,
        'cfr_iterations': 1500,
        'output': 'src/train_samples_dev',
        'championship_bet_sizing': True,
        'adaptive_cfr': False,
        'description': 'Development iteration (1-2 hours)',
    },
    'production': {
        'samples': 100000,
        'validation_samples': 20000,
        'cfr_iterations': 2500,
        'output': 'src/train_samples_production',
        'championship_bet_sizing': True,
        'adaptive_cfr': False,
        'description': 'Production quality (18-24 hours)',
    },
    'championship': {
        'samples': 500000,
        'validation_samples': 100000,
        'cfr_iterations': 2500,
        'output': 'src/train_samples_championship',
        'championship_bet_sizing': True,
        'adaptive_cfr': True,
        'description': 'Championship-level (4-5 days)',
    },
}


def estimate_time(samples: int, cfr_iters: int, adaptive: bool = False) -> str:
    """Estimate generation time based on parameters."""
    samples_per_sec = 2.5 * (2000 / cfr_iters)
    if adaptive:
        samples_per_sec *= 1.25  # 25% faster with adaptive CFR
    
    total_seconds = samples / samples_per_sec
    
    if total_seconds < 3600:
        minutes = int(total_seconds / 60)
        return f"~{minutes} minutes"
    elif total_seconds < 86400:
        hours = int(total_seconds / 3600)
        minutes = int((total_seconds % 3600) / 60)
        return f"~{hours}h {minutes}m"
    else:
        days = total_seconds / 86400
        return f"~{days:.1f} days"


def assess_quality(samples: int, cfr_iters: int) -> tuple:
    """Assess expected quality level and correlation."""
    if samples >= 500000 and cfr_iters >= 2500:
        return "Championship", ">0.85"
    elif samples >= 100000 and cfr_iters >= 2000:
        return "Production", "0.75-0.85"
    elif samples >= 50000 and cfr_iters >= 1500:
        return "Development", "0.65-0.75"
    elif samples >= 10000:
        return "Testing", "0.50-0.65"
    else:
        return "Insufficient", "<0.50"


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: Dict, output_path: str):
    """Save generation configuration for reproducibility."""
    config_file = Path(output_path) / 'generation_config.json'
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Configuration saved to {config_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified training data generation with profile support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Profiles:
  testing      : 1K samples, 500 CFR iters (~5-10 min)
  development  : 10K samples, 1500 CFR iters (~1-2 hours)
  production   : 100K samples, 2500 CFR iters (~18-24 hours)
  championship : 500K samples, 2500 CFR iters (~4-5 days)

Examples:
  # Quick test
  python scripts/generate_data.py --profile testing
  
  # Production with custom sample count
  python scripts/generate_data.py --profile production --samples 200000
  
  # Championship with adaptive CFR
  python scripts/generate_data.py --profile championship --adaptive-cfr
  
  # Custom configuration from file
  python scripts/generate_data.py --config config/data_generation/my_config.json
        """
    )
    
    # Profile or config selection
    parser.add_argument('--profile', type=str, choices=list(PROFILES.keys()),
                       help='Use predefined profile (testing/development/production/championship)')
    parser.add_argument('--config', type=str,
                       help='Load configuration from JSON file')
    
    # Parameter overrides
    parser.add_argument('--samples', type=int,
                       help='Number of training samples (overrides profile)')
    parser.add_argument('--validation-samples', type=int,
                       help='Number of validation samples (overrides profile)')
    parser.add_argument('--cfr-iters', type=int,
                       help='CFR iterations per sample (overrides profile)')
    parser.add_argument('--output', type=str,
                       help='Output directory (overrides profile)')
    parser.add_argument('--bucket-weights', type=str,
                       help='Path to bucket_weights.json for adaptive sampling')
    
    # Feature flags
    parser.add_argument('--championship-bet-sizing', action='store_true', default=None,
                       help='Enable per-street bet sizing (overrides profile)')
    parser.add_argument('--simple-bet-sizing', action='store_true',
                       help='Use simple pot-sized bets (overrides profile)')
    parser.add_argument('--adaptive-cfr', action='store_true', default=None,
                       help='Enable adaptive CFR iterations (overrides profile)')
    
    # Execution options
    parser.add_argument('--yes', '-y', action='store_true',
                       help='Skip confirmation prompt')
    parser.add_argument('--save-config', action='store_true',
                       help='Save configuration to output directory')
    
    args = parser.parse_args()
    
    # Determine configuration
    if args.config:
        config = load_config(args.config)
        source = f"config file: {args.config}"
    elif args.profile:
        config = PROFILES[args.profile].copy()
        source = f"profile: {args.profile}"
    else:
        # Default to development profile
        config = PROFILES['development'].copy()
        source = "profile: development (default)"
        print("⚠ No profile or config specified, using 'development' profile")
        print()
    
    # Apply parameter overrides
    if args.samples is not None:
        config['samples'] = args.samples
    if args.validation_samples is not None:
        config['validation_samples'] = args.validation_samples
    if args.cfr_iters is not None:
        config['cfr_iterations'] = args.cfr_iters
    if args.output is not None:
        config['output'] = args.output
    if args.championship_bet_sizing is not None:
        config['championship_bet_sizing'] = True
    if args.simple_bet_sizing:
        config['championship_bet_sizing'] = False
    if args.adaptive_cfr is not None:
        config['adaptive_cfr'] = True
    
    # Ensure validation samples default
    if 'validation_samples' not in config or config['validation_samples'] is None:
        config['validation_samples'] = max(1000, config['samples'] // 5)
    
    # Calculate estimates
    total_samples = config['samples'] + config['validation_samples']
    est_time = estimate_time(total_samples, config['cfr_iterations'], 
                            config.get('adaptive_cfr', False))
    quality, correlation = assess_quality(config['samples'], config['cfr_iterations'])
    
    # Display configuration
    print("="*70)
    print("UNIFIED TRAINING DATA GENERATION")
    print("="*70)
    print()
    print(f"Configuration source: {source}")
    if 'description' in config:
        print(f"Description: {config['description']}")
    print()
    print("Parameters:")
    print(f"  Training samples:      {config['samples']:,}")
    print(f"  Validation samples:    {config['validation_samples']:,}")
    print(f"  Total samples:         {total_samples:,}")
    print(f"  CFR iterations:        {config['cfr_iterations']}")
    print(f"  Championship bet sizing: {config.get('championship_bet_sizing', True)}")
    print(f"  Adaptive CFR:          {config.get('adaptive_cfr', False)}")
    print(f"  Output directory:      {config['output']}")
    print()
    print("Estimates:")
    print(f"  Generation time:       {est_time}")
    print(f"  Expected quality:      {quality}")
    print(f"  Expected correlation:  {correlation}")
    print()
    
    # Warnings
    warnings = []
    if config['samples'] < 10000:
        warnings.append(f"⚠ Low sample count ({config['samples']:,} < 10K minimum)")
    if config['cfr_iterations'] < 1500:
        warnings.append(f"⚠ Low CFR iterations ({config['cfr_iterations']} < 1500 minimum)")
    if quality in ['Insufficient', 'Testing']:
        warnings.append(f"⚠ Quality level '{quality}' - not recommended for production")
    
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  {w}")
        print()
    
    print("="*70)
    
    # Confirmation
    if not args.yes:
        response = input("\nProceed with data generation? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Aborted.")
            return
    
    print()
    print("Starting data generation...")
    print()
    
    # Load bucket weights if provided
    bucket_weights = None
    if args.bucket_weights and os.path.exists(args.bucket_weights):
        try:
            with open(args.bucket_weights, 'r') as f:
                weights = json.load(f)
                if len(weights) == 169:
                    bucket_weights = np.array(weights, dtype=np.float32)
                    print(f"✓ Loaded bucket weights from {args.bucket_weights}")
                    print(f"  Boosted buckets: {(bucket_weights > 1.5).sum()}")
                else:
                    print(f"⚠ Bucket weights file has wrong length ({len(weights)} != 169)")
        except Exception as e:
            print(f"⚠ Could not load bucket weights: {e}")
        print()
    
    # Save configuration if requested
    if args.save_config:
        save_config(config, config['output'])
        print()
    
    # Track time
    start_time = time.time()
    
    # Generate data
    try:
        generate_training_data(
            train_count=config['samples'],
            valid_count=config['validation_samples'],
            output_path=config['output'],
            cfr_iterations=config['cfr_iterations'],
            bucket_sampling_weights=bucket_weights,
            use_championship_bet_sizing=config.get('championship_bet_sizing', True),
            use_adaptive_cfr=config.get('adaptive_cfr', False)
        )
    except KeyboardInterrupt:
        print()
        print("="*70)
        print("⚠ Generation interrupted by user")
        print("="*70)
        return
    except Exception as e:
        print()
        print("="*70)
        print(f"❌ Error during generation: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return
    
    # Calculate actual time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time / 3600)
    minutes = int((elapsed_time % 3600) / 60)
    seconds = int(elapsed_time % 60)
    
    print()
    print("="*70)
    print("✓ DATA GENERATION COMPLETE")
    print("="*70)
    print()
    print(f"Total time: {hours}h {minutes}m {seconds}s")
    print(f"Data saved to: {config['output']}")
    print()
    print("Next steps:")
    print("  1. Validate: python scripts/validate_data.py")
    print("  2. Train: python scripts/train_model.py --profile production")
    print("  3. Validate model: python scripts/validate_model.py")
    print("="*70)


if __name__ == '__main__':
    main()
