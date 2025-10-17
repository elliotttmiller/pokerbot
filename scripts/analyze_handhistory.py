#!/usr/bin/env python3
"""
Analyze official DeepStack hand history data to extract insights for training optimization.

This script analyzes:
1. AIVAT analysis CSV files - extracting betting patterns, position effects, timing
2. LBR match logs - analyzing action sequences and outcomes
3. Statistical patterns that can inform training data generation

The insights will be used to improve:
- Range sampling strategies
- Bet sizing abstractions
- Street-specific training emphasis
- Position-aware data generation
"""

import os
import sys
import csv
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import numpy as np

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)


class HandHistoryAnalyzer:
    """Analyze official DeepStack hand history data."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.aivat_dir = self.data_dir / 'DeepStack_vs_IFP_pros' / 'AIVAT_analysis'
        self.lbr_dir = self.data_dir / 'vs_LBR' / 'deepstack'
        
        self.aivat_results = []
        self.lbr_results = []
        
    def analyze_aivat_data(self) -> Dict:
        """Analyze AIVAT CSV files for betting patterns and statistics."""
        print("Analyzing AIVAT data...")
        
        stats = {
            'total_hands': 0,
            'by_position': {'bb': 0, 'sb': 0},
            'by_street': {
                'preflop_only': 0,
                'to_flop': 0,
                'to_turn': 0,
                'to_river': 0
            },
            'bet_sizes': [],
            'raise_sizes': [],
            'avg_pot_sizes': [],
            'avg_hand_duration': [],
            'actions': defaultdict(int),
            'street_actions': {
                'preflop': defaultdict(int),
                'flop': defaultdict(int),
                'turn': defaultdict(int),
                'river': defaultdict(int)
            }
        }
        
        if not self.aivat_dir.exists():
            print(f"AIVAT directory not found: {self.aivat_dir}")
            return stats
            
        csv_files = list(self.aivat_dir.glob('results.*.csv'))
        print(f"Found {len(csv_files)} AIVAT CSV files")
        
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        stats['total_hands'] += 1
                        
                        # Position
                        pos = row.get('Position', '').strip().lower()
                        if pos in ['bb', 'sb']:
                            stats['by_position'][pos] += 1
                        
                        # Street progression
                        has_flop = bool(row.get('Flop Cards', '').strip())
                        has_turn = bool(row.get('Turn Card', '').strip())
                        has_river = bool(row.get('River Card', '').strip())
                        
                        if has_river:
                            stats['by_street']['to_river'] += 1
                        elif has_turn:
                            stats['by_street']['to_turn'] += 1
                        elif has_flop:
                            stats['by_street']['to_flop'] += 1
                        else:
                            stats['by_street']['preflop_only'] += 1
                        
                        # Parse actions
                        preflop = row.get('Pre-flop', '').strip()
                        flop = row.get('Flop', '').strip()
                        turn = row.get('Turn', '').strip()
                        river = row.get('River', '').strip()
                        
                        self._parse_actions(preflop, stats['street_actions']['preflop'], stats)
                        if flop:
                            self._parse_actions(flop, stats['street_actions']['flop'], stats)
                        if turn:
                            self._parse_actions(turn, stats['street_actions']['turn'], stats)
                        if river:
                            self._parse_actions(river, stats['street_actions']['river'], stats)
                        
                        # Duration
                        try:
                            duration = float(row.get('Total Seconds', 0))
                            if duration > 0:
                                stats['avg_hand_duration'].append(duration)
                        except:
                            pass
                            
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
        
        return stats
    
    def _parse_actions(self, action_str: str, street_stats: Dict, global_stats: Dict):
        """Parse action string and update statistics."""
        if not action_str:
            return
            
        # Actions: c=call/check, f=fold, rX=raise to X
        actions = action_str.lower().replace(' ', '')
        
        i = 0
        while i < len(actions):
            if actions[i] == 'c':
                street_stats['call'] += 1
                global_stats['actions']['call'] += 1
                i += 1
            elif actions[i] == 'f':
                street_stats['fold'] += 1
                global_stats['actions']['fold'] += 1
                i += 1
            elif actions[i] == 'r':
                # Extract raise size
                j = i + 1
                while j < len(actions) and (actions[j].isdigit() or actions[j] == '.'):
                    j += 1
                if j > i + 1:
                    try:
                        size = float(actions[i+1:j])
                        global_stats['raise_sizes'].append(size)
                    except:
                        pass
                street_stats['raise'] += 1
                global_stats['actions']['raise'] += 1
                i = j
            else:
                i += 1
    
    def analyze_lbr_data(self) -> Dict:
        """Analyze LBR match logs for action patterns."""
        print("Analyzing LBR data...")
        
        stats = {
            'total_hands': 0,
            'outcomes': defaultdict(int),
            'avg_pots': [],
            'hand_patterns': defaultdict(int)
        }
        
        if not self.lbr_dir.exists():
            print(f"LBR directory not found: {self.lbr_dir}")
            return stats
            
        lbr_files = list(self.lbr_dir.glob('lbr_*_ds_*.out'))
        print(f"Found {len(lbr_files)} LBR log files")
        
        # Sample a subset for analysis (too many files)
        import random
        sample_files = random.sample(lbr_files, min(10, len(lbr_files)))
        
        for lbr_file in sample_files:
            try:
                with open(lbr_file, 'r') as f:
                    for line in f:
                        if ':' not in line or line.startswith('In config') or line.startswith('Using seed'):
                            continue
                        
                        stats['total_hands'] += 1
                        
                        # Parse hand: "1:BR,Player:0r200c/cr300c/r1493f:KhKc,6h9c|/2s5dTs/2d:600,-600(600.000000)"
                        parts = line.strip().split(':')
                        if len(parts) >= 5:
                            # Action sequence
                            action_seq = parts[2]
                            # Outcome
                            outcome_str = parts[4]
                            try:
                                outcome_match = re.search(r'([-\d]+),([-\d]+)', outcome_str)
                                if outcome_match:
                                    p1_chips = int(outcome_match.group(1))
                                    pot = abs(p1_chips)
                                    stats['avg_pots'].append(pot)
                                    
                                    if p1_chips > 0:
                                        stats['outcomes']['p1_win'] += 1
                                    elif p1_chips < 0:
                                        stats['outcomes']['p2_win'] += 1
                                    else:
                                        stats['outcomes']['tie'] += 1
                            except:
                                pass
                                
            except Exception as e:
                print(f"Error processing {lbr_file}: {e}")
        
        return stats
    
    def generate_insights(self, aivat_stats: Dict, lbr_stats: Dict) -> Dict:
        """Generate training insights from analyzed data."""
        insights = {
            'street_distribution': {},
            'recommended_cfr_iterations': None,
            'bet_sizing_abstraction': [],
            'position_importance': {},
            'training_recommendations': []
        }
        
        # Street distribution insights
        total = sum(aivat_stats['by_street'].values())
        if total > 0:
            insights['street_distribution'] = {
                'preflop': aivat_stats['by_street']['preflop_only'] / total,
                'flop': aivat_stats['by_street']['to_flop'] / total,
                'turn': aivat_stats['by_street']['to_turn'] / total,
                'river': aivat_stats['by_street']['to_river'] / total
            }
            
            # Recommendation: emphasize postflop (as in current implementation)
            postflop_pct = (total - aivat_stats['by_street']['preflop_only']) / total
            if postflop_pct > 0.6:
                insights['training_recommendations'].append(
                    f"Emphasize postflop training: {postflop_pct*100:.1f}% of hands reach flop or beyond"
                )
        
        # Bet sizing analysis
        if aivat_stats['raise_sizes']:
            raise_sizes = np.array(aivat_stats['raise_sizes'])
            percentiles = np.percentile(raise_sizes, [25, 50, 75, 90, 95])
            insights['bet_sizing_abstraction'] = [
                {'percentile': 25, 'size': percentiles[0]},
                {'percentile': 50, 'size': percentiles[1]},
                {'percentile': 75, 'size': percentiles[2]},
                {'percentile': 90, 'size': percentiles[3]},
                {'percentile': 95, 'size': percentiles[4]}
            ]
            
            # Common bet sizes in real play
            common_sizes = [0.5, 0.75, 1.0, 1.5, 2.0]  # Pot-relative
            insights['training_recommendations'].append(
                f"Use bet abstraction with pot-relative sizes: {common_sizes}"
            )
        
        # Position importance
        total_pos = sum(aivat_stats['by_position'].values())
        if total_pos > 0:
            insights['position_importance'] = {
                pos: count / total_pos 
                for pos, count in aivat_stats['by_position'].items()
            }
        
        # CFR iterations recommendation
        # Based on DeepStack paper and current best practices
        insights['recommended_cfr_iterations'] = {
            'minimum': 1500,
            'recommended': 2000,
            'championship': 2500,
            'rationale': 'DeepStack paper uses 1000+ iterations; modern implementations use 2000-2500 for better convergence'
        }
        
        # Action frequency insights
        if aivat_stats['actions']:
            total_actions = sum(aivat_stats['actions'].values())
            action_dist = {
                action: count / total_actions 
                for action, count in aivat_stats['actions'].items()
            }
            insights['action_distribution'] = action_dist
        
        return insights
    
    def run_full_analysis(self) -> Dict:
        """Run complete analysis and return results."""
        print("="*70)
        print("DeepStack Hand History Analysis")
        print("="*70)
        print()
        
        aivat_stats = self.analyze_aivat_data()
        print(f"\nAIVAT Analysis: {aivat_stats['total_hands']} hands analyzed")
        
        lbr_stats = self.analyze_lbr_data()
        print(f"LBR Analysis: {lbr_stats['total_hands']} hands analyzed")
        print()
        
        insights = self.generate_insights(aivat_stats, lbr_stats)
        
        return {
            'aivat_statistics': aivat_stats,
            'lbr_statistics': lbr_stats,
            'insights': insights
        }


def main():
    """Main analysis function."""
    data_dir = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'data', 
        'official_deepstack_handhistory'
    )
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    analyzer = HandHistoryAnalyzer(data_dir)
    results = analyzer.run_full_analysis()
    
    # Print insights
    print("="*70)
    print("INSIGHTS FOR TRAINING OPTIMIZATION")
    print("="*70)
    print()
    
    insights = results['insights']
    
    if 'street_distribution' in insights and insights['street_distribution']:
        print("Street Distribution in Real Play:")
        for street, pct in insights['street_distribution'].items():
            print(f"  {street:10s}: {pct*100:5.1f}%")
        print()
    
    if 'recommended_cfr_iterations' in insights:
        cfr = insights['recommended_cfr_iterations']
        print("CFR Iterations Recommendation:")
        print(f"  Minimum: {cfr['minimum']}")
        print(f"  Recommended: {cfr['recommended']}")
        print(f"  Championship: {cfr['championship']}")
        print(f"  Rationale: {cfr['rationale']}")
        print()
    
    if 'bet_sizing_abstraction' in insights and insights['bet_sizing_abstraction']:
        print("Bet Sizing Analysis:")
        for item in insights['bet_sizing_abstraction']:
            print(f"  {item['percentile']}th percentile: {item['size']:.2f}")
        print()
    
    if insights['training_recommendations']:
        print("Training Recommendations:")
        for i, rec in enumerate(insights['training_recommendations'], 1):
            print(f"  {i}. {rec}")
        print()
    
    # Save results
    output_file = os.path.join(
        os.path.dirname(__file__),
        '..',
        'data',
        'handhistory_analysis.json'
    )
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Full analysis saved to: {output_file}")
    print()
    print("="*70)
    print("Analysis Complete")
    print("="*70)


if __name__ == '__main__':
    main()
