"""
COCO Dataset Loader for Poker Vision Training

Loads COCO-formatted annotations for fine-tuning QWEN 2.5-7B VL on poker screenshots.
"""

import json
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PokerAnnotation:
    """Annotation for a single poker screenshot."""
    image_id: int
    image_path: str
    width: int
    height: int
    game_state: Dict[str, Any]


class COCOPokerDataset:
    """
    COCO-formatted dataset loader for poker screenshots.
    
    Supports streaming mode for memory-efficient training in Colab.
    
    Usage:
        dataset = COCOPokerDataset("annotations.coco.json", "images/")
        for sample in dataset:
            print(sample.image_path, sample.game_state)
    """
    
    def __init__(
        self,
        annotations_path: str,
        images_dir: str,
        streaming: bool = True
    ):
        """
        Initialize dataset loader.
        
        Args:
            annotations_path: Path to COCO annotations JSON
            images_dir: Directory containing images
            streaming: If True, load annotations lazily for memory efficiency
        """
        self.annotations_path = Path(annotations_path)
        self.images_dir = Path(images_dir)
        self.streaming = streaming
        
        self._annotations = None
        self._images = None
        self._categories = None
        
        if not streaming:
            self._load_annotations()
    
    def _load_annotations(self):
        """Load all annotations into memory."""
        with open(self.annotations_path, 'r') as f:
            data = json.load(f)
        
        self._images = {img['id']: img for img in data.get('images', [])}
        self._annotations = data.get('annotations', [])
        self._categories = {cat['id']: cat for cat in data.get('categories', [])}
    
    def __len__(self) -> int:
        """Return number of samples."""
        if self._annotations is None:
            self._load_annotations()
        return len(self._annotations)
    
    def __iter__(self) -> Iterator[PokerAnnotation]:
        """Iterate over samples."""
        if self.streaming:
            yield from self._stream_annotations()
        else:
            if self._annotations is None:
                self._load_annotations()
            for ann in self._annotations:
                yield self._process_annotation(ann)
    
    def _stream_annotations(self) -> Iterator[PokerAnnotation]:
        """Stream annotations without loading all into memory."""
        with open(self.annotations_path, 'r') as f:
            data = json.load(f)
        
        images = {img['id']: img for img in data.get('images', [])}
        
        for ann in data.get('annotations', []):
            image_info = images.get(ann['image_id'], {})
            yield PokerAnnotation(
                image_id=ann['image_id'],
                image_path=str(self.images_dir / image_info.get('file_name', '')),
                width=image_info.get('width', 0),
                height=image_info.get('height', 0),
                game_state=self._extract_game_state(ann)
            )
    
    def _process_annotation(self, ann: Dict) -> PokerAnnotation:
        """Process a single annotation."""
        image_info = self._images.get(ann['image_id'], {})
        return PokerAnnotation(
            image_id=ann['image_id'],
            image_path=str(self.images_dir / image_info.get('file_name', '')),
            width=image_info.get('width', 0),
            height=image_info.get('height', 0),
            game_state=self._extract_game_state(ann)
        )
    
    def _extract_game_state(self, ann: Dict) -> Dict[str, Any]:
        """Extract game state from annotation."""
        # Support multiple annotation formats
        if 'game_state' in ann:
            return ann['game_state']
        
        # Build game state from individual fields
        return {
            'hole_cards': ann.get('hole_cards', []),
            'community_cards': ann.get('community_cards', []),
            'pot_size': ann.get('pot_size', 0),
            'current_bet': ann.get('current_bet', 0),
            'player_stack': ann.get('player_stack', 1000),
            'opponent_stack': ann.get('opponent_stack', 1000),
            'street': ann.get('street', 'preflop'),
            'action_required': ann.get('action_required', True),
            'available_actions': ann.get('available_actions', ['fold', 'call', 'raise'])
        }
    
    def get_sample(self, index: int) -> PokerAnnotation:
        """Get a specific sample by index."""
        if self._annotations is None:
            self._load_annotations()
        return self._process_annotation(self._annotations[index])
    
    def to_training_format(self) -> List[Dict[str, Any]]:
        """
        Convert dataset to Unsloth training format.
        
        Returns:
            List of dictionaries with 'image', 'prompt', 'response' keys
        """
        training_samples = []
        
        for sample in self:
            response_json = json.dumps(sample.game_state, indent=2)
            training_samples.append({
                'image': sample.image_path,
                'prompt': self._get_detection_prompt(),
                'response': response_json
            })
        
        return training_samples
    
    def _get_detection_prompt(self) -> str:
        """Return the detection prompt for training."""
        return """Analyze this poker game screenshot and extract the exact game state.
Return ONLY a JSON object with:
- hole_cards: List of player's cards (e.g., ["As", "Kh"])
- community_cards: List of board cards
- pot_size: Integer pot amount
- current_bet: Integer bet to call
- player_stack: Player's chip count
- opponent_stack: Opponent's chip count
- street: "preflop", "flop", "turn", or "river"
- action_required: true/false
- available_actions: List of valid actions"""


def load_coco_dataset(
    annotations_path: str,
    images_dir: str,
    streaming: bool = True
) -> COCOPokerDataset:
    """
    Convenience function to load COCO poker dataset.
    
    Args:
        annotations_path: Path to annotations JSON
        images_dir: Path to images directory
        streaming: Enable streaming for memory efficiency
        
    Returns:
        COCOPokerDataset instance
    """
    return COCOPokerDataset(annotations_path, images_dir, streaming)


__all__ = ['COCOPokerDataset', 'PokerAnnotation', 'load_coco_dataset']
