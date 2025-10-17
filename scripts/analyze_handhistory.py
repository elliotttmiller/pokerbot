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
from typing import Dict, List, Tuple, Optional
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
        # Accumulators for experimental pot-relative ratios per street (0=pre,1=flop,2=turn,3=river)
        self._pot_rel_raises = {0: [], 1: [], 2: [], 3: []}

    @staticmethod
    def _parse_tokens(action_str: str) -> List[Tuple[str, Optional[float]]]:
        """Parse compact action string like 'cr300c' into tokens [('c',None), ('r',300.0), ('c',None)]."""
        tokens: List[Tuple[str, Optional[float]]] = []
        if not action_str:
            return tokens
        s = action_str.strip().lower().replace(' ', '')
        i = 0
        while i < len(s):
            ch = s[i]
            if ch in ('c', 'f'):
                tokens.append((ch, None))
                i += 1
            elif ch == 'r':
                j = i + 1
                while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                    j += 1
                amt = None
                if j > i + 1:
                    try:
                        amt = float(s[i+1:j])
                    except:
                        amt = None
                tokens.append(('r', amt))
                i = j
            elif ch.isdigit():
                # Some logs prefix street with actor index (e.g., '0r200c') - skip leading digits
                while i < len(s) and s[i].isdigit():
                    i += 1
            else:
                i += 1
        return tokens

    @staticmethod
    def _compute_pot_relative_from_streets(preflop: str, flop: str, turn: str, river: str,
                                           sb: int = 50, bb: int = 100) -> Dict[int, List[float]]:
        """
        Experimental reconstruction of pot-relative raise ratios per street.
        Assumptions: heads-up, blinds sb/bb, 'rX' means raise to X chips on that street.
        Returns mapping street_index -> list of ratios (bet_increment / pot_before_bet).
        """
        ratios = {0: [], 1: [], 2: [], 3: []}
        # Helper to process a street
        def process_street(tokens: List[Tuple[str, Optional[float]]], street_idx: int,
                           pot: float, contrib: List[float]) -> Tuple[float, List[float]]:
            current_bet = max(contrib) if contrib else 0.0
            actor = 0  # actor index toggle; exact identity doesn't matter for ratios
            for kind, amt in tokens:
                if kind == 'r' and amt is not None:
                    pot_before = pot if pot > 0 else 1.0  # avoid div by zero
                    to_put = max(0.0, amt - contrib[actor])
                    ratio = to_put / pot_before
                    if 0.0 < ratio < 5.0:  # discard pathological values
                        ratios[street_idx].append(ratio)
                    contrib[actor] += to_put
                    current_bet = max(current_bet, contrib[actor])
                    pot += to_put
                    actor = 1 - actor
                elif kind == 'c':
                    to_put = max(0.0, current_bet - contrib[actor])
                    contrib[actor] += to_put
                    pot += to_put
                    actor = 1 - actor
                elif kind == 'f':
                    break
            return pot, contrib

        # Initialize preflop with blinds
        pot = float(sb + bb)
        contrib = [float(sb), float(bb)]
        # Preflop
        pre_tokens = HandHistoryAnalyzer._parse_tokens(preflop)
        pot, contrib = process_street(pre_tokens, 0, pot, contrib)
        # Street transition -> carry pot, reset contributions
        contrib = [0.0, 0.0]
        # Flop
        flop_tokens = HandHistoryAnalyzer._parse_tokens(flop)
        pot, contrib = process_street(flop_tokens, 1, pot, contrib)
        contrib = [0.0, 0.0]
        # Turn
        turn_tokens = HandHistoryAnalyzer._parse_tokens(turn)
        pot, contrib = process_street(turn_tokens, 2, pot, contrib)
        contrib = [0.0, 0.0]
        # River
        river_tokens = HandHistoryAnalyzer._parse_tokens(river)
        pot, contrib = process_street(river_tokens, 3, pot, contrib)
        return ratios
        
    @staticmethod
    def _truthy_cell(val: Optional[str]) -> bool:
        """Return True if a CSV cell contains meaningful content (not empty or placeholder)."""
        if val is None:
            return False
        s = str(val).strip().lower()
        return s not in ("", "-", "na", "n/a", "none", "null")

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
                # Try multiple encodings for robustness
                encodings = ['utf-8-sig', 'utf-8', 'latin-1']
                success = False
                for enc in encodings:
                    try:
                        with open(csv_file, 'r', encoding=enc, newline='') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                stats['total_hands'] += 1

                                # Position
                                pos = row.get('Position', '')
                                pos = pos.strip().lower() if isinstance(pos, str) else ''
                                if pos in ['bb', 'sb']:
                                    stats['by_position'][pos] += 1

                                # Card columns (header variants)
                                flop_cards = row.get('Flop Cards') or row.get('FlopCards') or row.get('Flop')
                                turn_card = row.get('Turn Card') or row.get('TurnCard') or row.get('Turn')
                                river_card = row.get('River Card') or row.get('RiverCard') or row.get('River')

                                # Action columns (header variants)
                                preflop = (row.get('Pre-flop') or row.get('Preflop') or '')
                                flop = (row.get('Flop') or '')
                                turn = (row.get('Turn') or '')
                                river = (row.get('River') or '')
                                preflop = preflop.strip() if isinstance(preflop, str) else ''
                                flop = flop.strip() if isinstance(flop, str) else ''
                                turn = turn.strip() if isinstance(turn, str) else ''
                                river = river.strip() if isinstance(river, str) else ''

                                # Final reach flags: OR of cards/actions, count ONCE per hand
                                has_flop = self._truthy_cell(flop_cards) or self._truthy_cell(flop)
                                has_turn = self._truthy_cell(turn_card) or self._truthy_cell(turn)
                                has_river = self._truthy_cell(river_card) or self._truthy_cell(river)

                                if has_river:
                                    stats['by_street']['to_river'] += 1
                                elif has_turn:
                                    stats['by_street']['to_turn'] += 1
                                elif has_flop:
                                    stats['by_street']['to_flop'] += 1
                                else:
                                    stats['by_street']['preflop_only'] += 1

                                # Parse actions for raise sizes and action distribution
                                self._parse_actions(preflop, stats['street_actions']['preflop'], stats)
                                if flop:
                                    self._parse_actions(flop, stats['street_actions']['flop'], stats)
                                if turn:
                                    self._parse_actions(turn, stats['street_actions']['turn'], stats)
                                if river:
                                    self._parse_actions(river, stats['street_actions']['river'], stats)

                                # Experimental: collect pot-relative ratios per street
                                try:
                                    ratios = self._compute_pot_relative_from_streets(preflop, flop, turn, river)
                                    for si in (0, 1, 2, 3):
                                        if ratios.get(si):
                                            self._pot_rel_raises[si].extend(ratios[si])
                                except Exception:
                                    pass

                                # Duration
                                try:
                                    # Some dumps use 'Total Seconds' or 'TotalSeconds'
                                    duration = float((row.get('Total Seconds') or row.get('TotalSeconds') or 0))
                                    if duration > 0:
                                        stats['avg_hand_duration'].append(duration)
                                except:
                                    pass
                        success = True
                        break
                    except Exception:
                        continue
                if not success:
                    raise RuntimeError(f"Failed to parse {csv_file} with known encodings")
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
    
    def analyze_lbr_data(self, max_files: int = 20, analyze_all: bool = False) -> Dict:
        """Analyze LBR match logs for action patterns."""
        print("Analyzing LBR data...")
        
        stats = {
            'total_hands': 0,
            'outcomes': defaultdict(int),
            'avg_pots': [],
            'hand_patterns': defaultdict(int),
            'by_street': {
                'preflop_only': 0,
                'to_flop': 0,
                'to_turn': 0,
                'to_river': 0
            },
            'raise_sizes': []
        }
        
        if not self.lbr_dir.exists():
            print(f"LBR directory not found: {self.lbr_dir}")
            return stats
            
        lbr_files = list(self.lbr_dir.glob('lbr_*_ds_*.out'))
        print(f"Found {len(lbr_files)} LBR log files")
        
        # Sample a subset for analysis (too many files)
        import random
        if analyze_all:
            sample_files = lbr_files
        else:
            sample_files = random.sample(lbr_files, min(max_files, len(lbr_files)))
        
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
                            # Action sequence and board segment
                            action_seq = parts[2]
                            board_part = parts[3] if len(parts) > 3 else ''
                            # Outcome
                            outcome_str = parts[4]

                            # Determine street reach using board markers after '|'
                            board_seg = board_part.split('|', 1)[-1] if '|' in board_part else ''
                            nonempty = [t for t in board_seg.split('/') if t]
                            n = len(nonempty)
                            has_flop = n >= 1
                            has_turn = n >= 2
                            has_river = n >= 3
                            if has_river:
                                stats['by_street']['to_river'] += 1
                            elif has_turn:
                                stats['by_street']['to_turn'] += 1
                            elif has_flop:
                                stats['by_street']['to_flop'] += 1
                            else:
                                stats['by_street']['preflop_only'] += 1

                            # Raise sizes (absolute amounts)
                            for m in re.finditer(r'r(\d+(?:\.\d+)?)', action_seq):
                                try:
                                    stats['raise_sizes'].append(float(m.group(1)))
                                except:
                                    pass

                            # Experimental: per-street pot-relative ratios
                            try:
                                streets = action_seq.split('/')
                                pf = streets[0] if len(streets) > 0 else ''
                                fl = streets[1] if len(streets) > 1 else ''
                                tr = streets[2] if len(streets) > 2 else ''
                                rv = streets[3] if len(streets) > 3 else ''
                                ratios = self._compute_pot_relative_from_streets(pf, fl, tr, rv)
                                for si in (0, 1, 2, 3):
                                    if ratios.get(si):
                                        self._pot_rel_raises[si].extend(ratios[si])
                            except Exception:
                                pass

                            try:
                                outcome_match = re.search(r'([\-\d]+),([\-\d]+)', outcome_str)
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
        
        # Street distribution insights (blend AIVAT + LBR if available)
        a_total = sum(aivat_stats['by_street'].values()) if aivat_stats and 'by_street' in aivat_stats else 0
        l_total = sum(lbr_stats['by_street'].values()) if lbr_stats and 'by_street' in lbr_stats else 0
        combined = {'preflop_only': 0, 'to_flop': 0, 'to_turn': 0, 'to_river': 0}
        if a_total:
            for k in combined:
                combined[k] += aivat_stats['by_street'].get(k, 0)
        if l_total:
            for k in combined:
                combined[k] += lbr_stats['by_street'].get(k, 0)
        c_total = sum(combined.values())
        if c_total > 0:
            insights['street_distribution'] = {
                'preflop': combined['preflop_only'] / c_total,
                'flop': combined['to_flop'] / c_total,
                'turn': combined['to_turn'] / c_total,
                'river': combined['to_river'] / c_total
            }
            postflop_pct = (c_total - combined['preflop_only']) / c_total
            if postflop_pct > 0.6:
                insights['training_recommendations'].append(
                    f"Emphasize postflop training: {postflop_pct*100:.1f}% of hands reach flop or beyond"
                )
        
        # Bet sizing analysis (combine raise sizes from AIVAT + LBR; absolute not pot-relative)
        combined_raises = []
        if aivat_stats.get('raise_sizes'):
            combined_raises.extend(aivat_stats['raise_sizes'])
        if lbr_stats.get('raise_sizes'):
            combined_raises.extend(lbr_stats['raise_sizes'])
        if combined_raises:
            raise_sizes = np.array(combined_raises)
            percentiles = np.percentile(raise_sizes, [25, 50, 75, 90, 95])
            insights['bet_sizing_abstraction'] = [
                {'percentile': 25, 'size': percentiles[0]},
                {'percentile': 50, 'size': percentiles[1]},
                {'percentile': 75, 'size': percentiles[2]},
                {'percentile': 90, 'size': percentiles[3]},
                {'percentile': 95, 'size': percentiles[4]}
            ]
            # Note: prefer pot-relative sizes once pot reconstruction is implemented
            insights['training_recommendations'].append(
                "Consider refining bet sizing to pot-relative ratios once pot reconstruction is implemented"
            )

        # Experimental: pot-relative bet sizing per street using reconstructed ratios
        try:
            per_street = {}
            for si in (1, 2, 3):  # postflop only for override
                arr = np.array(self._pot_rel_raises.get(si, []), dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size >= 20:
                    qs = np.percentile(arr, [25, 50, 75, 90])
                    # Clip to reasonable range and round to 2 decimals
                    qs = np.clip(qs, 0.1, 2.5)
                    per_street[si] = [float(round(x, 2)) for x in qs]
            if per_street:
                insights['bet_sizing_pot_relative'] = per_street
                insights['training_recommendations'].append(
                    'Derived experimental pot-relative bet sizing from reconstructed pot dynamics'
                )
        except Exception:
            pass
        
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
    
    def run_full_analysis(self, max_lbr_files: int = 20, analyze_all_lbr: bool = False) -> Dict:
        """Run complete analysis and return results."""
        print("="*70)
        print("DeepStack Hand History Analysis")
        print("="*70)
        print()
        
        aivat_stats = self.analyze_aivat_data()
        print(f"\nAIVAT Analysis: {aivat_stats['total_hands']} hands analyzed")
        
        lbr_stats = self.analyze_lbr_data(max_files=max_lbr_files, analyze_all=analyze_all_lbr)
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
    import argparse
    parser = argparse.ArgumentParser(description='Analyze DeepStack hand history data')
    parser.add_argument('--data-dir', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'data', 'official_deepstack_handhistory'))
    parser.add_argument('--export-config', action='store_true', help='Export derived parameters to config/data_generation/parameters')
    parser.add_argument('--cap-streets', action='store_true', help='When exporting, cap street_distribution with minimum per postflop street and renormalize')
    parser.add_argument('--min-postflop-pct', type=float, default=0.05, help='Minimum proportion for each of flop/turn/river when capping (default 0.05)')
    parser.add_argument('--smoothing-alpha', type=float, default=None, help='Optional smoothing_alpha to include in exported config')
    parser.add_argument('--pot-relative-bets', action='store_true', help='Experimental: attempt pot reconstruction to export per-street bet_sizing_override as pot-relative ratios')
    parser.add_argument('--max-lbr-files', type=int, default=20, help='Max LBR logs to sample (ignored if --analyze-all-lbr)')
    parser.add_argument('--analyze-all-lbr', action='store_true', help='Analyze all LBR logs instead of a sample')
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    analyzer = HandHistoryAnalyzer(data_dir)
    results = analyzer.run_full_analysis(max_lbr_files=args.max_lbr_files, analyze_all_lbr=args.analyze_all_lbr)
    
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

    # Optionally export derived parameters for data generation
    if args.export_config:
        from datetime import datetime
        export_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'data_generation', 'parameters'))
        os.makedirs(export_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        export_path = os.path.join(export_dir, f'analytics_{ts}.json')

        insights = results.get('insights', {})
        street_dist = dict(insights.get('street_distribution', {}) or {})
        # Optionally cap street distribution extremes
        if args.cap_streets and street_dist:
            try:
                pre = float(street_dist.get('preflop', 0.25))
                flop = float(street_dist.get('flop', 0.35))
                turn = float(street_dist.get('turn', 0.20))
                river = float(street_dist.get('river', 0.20))
                min_pf = 0.0  # don't enforce min on preflop
                min_pp = max(0.0, float(args.min_postflop_pct))
                flop = max(flop, min_pp)
                turn = max(turn, min_pp)
                river = max(river, min_pp)
                # Renormalize while keeping relative proportions
                s = pre + flop + turn + river
                if s > 0:
                    pre, flop, turn, river = pre/s, flop/s, turn/s, river/s
                street_dist = {'preflop': pre, 'flop': flop, 'turn': turn, 'river': river}
            except Exception:
                pass
        cfr = insights.get('recommended_cfr_iterations', {})
        payload = {
            'name': f'analytics_{ts}',
            'source': 'official_deepstack_handhistory',
            'preflop_weight': float(street_dist.get('preflop', 0.25)) if street_dist else 0.25,
            'flop_weight': float(street_dist.get('flop', 0.35)) if street_dist else 0.35,
            'turn_weight': float(street_dist.get('turn', 0.20)) if street_dist else 0.20,
            'river_weight': float(street_dist.get('river', 0.20)) if street_dist else 0.20,
            'cfr_iterations': int(cfr.get('recommended', 2000)) if cfr else 2000,
            'bet_sizing_recommendation': insights.get('bet_sizing_abstraction', []),
            'notes': 'Derived from analyzed official hand history data'
        }
        # Export full dict too, for convenience
        if street_dist:
            payload['street_distribution'] = street_dist
        # Include smoothing_alpha if provided
        if args.smoothing_alpha is not None:
            try:
                payload['smoothing_alpha'] = float(args.smoothing_alpha)
            except Exception:
                pass
        # Optional: experimental pot-relative bet sizing reconstruction placeholder
        if args.pot_relative_bets:
            try:
                # Placeholder export structure; real reconstruction to be implemented
                payload.setdefault('bet_sizing_override', {})
                payload['bet_sizing_override'].update({
                    # Streets by index for generator: 0=pre,1=flop,2=turn,3=river
                    # Fill with sample ratios as placeholders to be replaced by proper reconstruction
                    1: [0.33, 0.5, 0.75, 1.0],
                    2: [0.5, 0.75, 1.0],
                    3: [0.66, 1.0, 1.5]
                })
                payload.setdefault('notes', '')
                payload['notes'] += ' | Includes experimental pot-relative bet sizing placeholders'
            except Exception:
                pass
        with open(export_path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"Derived config exported to: {export_path}")
    print()
    print("="*70)
    print("Analysis Complete")
    print("="*70)


if __name__ == '__main__':
    main()
