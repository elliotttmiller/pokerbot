"""
Multi-Modal Vision Detector for QWEN 2.5-7B VL Integration.

This module provides the interface for using a fine-tuned QWEN 2.5-7B Vision-Language
model to detect poker game state from screenshots. It is designed to be compatible
with the Unsloth training pipeline.

Architecture References:
- DeepStack.pdf: Game state representation
- self-operating-computer: Vision/multimodal patterns
- gto-poker-bot: GTO-compatible state extraction
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Optional imports for vision model
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class DetectedCard:
    """Represents a detected playing card."""
    rank: str  # 2-9, T, J, Q, K, A
    suit: str  # s, h, d, c
    confidence: float = 1.0
    
    @classmethod
    def from_string(cls, card_str: str) -> 'DetectedCard':
        """Parse card from string notation (e.g., 'As', 'Kh', 'Td')."""
        if len(card_str) < 2:
            raise ValueError(f"Invalid card string: {card_str}")
        
        rank = card_str[:-1].upper()
        suit = card_str[-1].lower()
        
        # Normalize rank
        if rank == '10':
            rank = 'T'
        
        valid_ranks = '23456789TJQKA'
        valid_suits = 'shdc'
        
        if rank not in valid_ranks:
            raise ValueError(f"Invalid rank: {rank}")
        if suit not in valid_suits:
            raise ValueError(f"Invalid suit: {suit}")
        
        return cls(rank=rank, suit=suit)
    
    def to_string(self) -> str:
        """Convert to string notation."""
        return f"{self.rank}{self.suit}"
    
    def __str__(self) -> str:
        return self.to_string()


@dataclass
class DetectedGameState:
    """
    Represents the complete detected game state.
    
    Attributes:
        hole_cards: Player's private cards
        community_cards: Board cards
        pot_size: Current pot amount
        current_bet: Amount to call
        player_stack: Player's remaining chips
        opponent_stack: Opponent's remaining chips
        street: Current betting round (preflop, flop, turn, river)
        action_required: Whether player needs to act
        available_actions: List of valid actions
        confidence: Overall detection confidence
        raw_response: Raw model output for debugging
    """
    hole_cards: List[DetectedCard]
    community_cards: List[DetectedCard]
    pot_size: int
    current_bet: int
    player_stack: int
    opponent_stack: int
    street: str
    action_required: bool
    available_actions: List[str]
    confidence: float = 1.0
    raw_response: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'hole_cards': [c.to_string() for c in self.hole_cards],
            'community_cards': [c.to_string() for c in self.community_cards],
            'pot_size': self.pot_size,
            'current_bet': self.current_bet,
            'player_stack': self.player_stack,
            'opponent_stack': self.opponent_stack,
            'street': self.street,
            'action_required': self.action_required,
            'available_actions': self.available_actions,
            'confidence': self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DetectedGameState':
        """Create from dictionary."""
        return cls(
            hole_cards=[DetectedCard.from_string(c) for c in data.get('hole_cards', [])],
            community_cards=[DetectedCard.from_string(c) for c in data.get('community_cards', [])],
            pot_size=data.get('pot_size', 0),
            current_bet=data.get('current_bet', 0),
            player_stack=data.get('player_stack', 1000),
            opponent_stack=data.get('opponent_stack', 1000),
            street=data.get('street', 'preflop'),
            action_required=data.get('action_required', True),
            available_actions=data.get('available_actions', ['fold', 'call', 'raise']),
            confidence=data.get('confidence', 1.0),
            raw_response=data.get('raw_response')
        )


class MultiModalVisionDetector:
    """
    Vision detector using QWEN 2.5-7B VL for poker game state detection.
    
    This class provides a unified interface for vision-based game state detection,
    supporting both local models (via transformers/Unsloth) and API-based fallback.
    
    Usage:
        detector = MultiModalVisionDetector(model_path="models/qwen_poker_vl")
        state = detector.detect("screenshot.png")
        print(state.hole_cards, state.pot_size)
    """
    
    # Detection prompt template optimized for QWEN 2.5-7B VL
    DETECTION_PROMPT = """Analyze this poker game screenshot and extract the exact game state.

INSTRUCTIONS:
1. Identify all visible cards (player's hole cards and community cards on the board)
2. Read the pot size and any bet amounts shown
3. Determine the current betting street based on community cards:
   - preflop: no community cards
   - flop: 3 community cards
   - turn: 4 community cards
   - river: 5 community cards
4. Check if player action is required (action buttons visible)

OUTPUT FORMAT (JSON only, no other text):
{
    "hole_cards": ["As", "Kh"],
    "community_cards": ["Qd", "Jc", "Ts"],
    "pot_size": 150,
    "current_bet": 50,
    "player_stack": 1000,
    "opponent_stack": 1200,
    "street": "flop",
    "action_required": true,
    "available_actions": ["fold", "call", "raise"]
}

CARD NOTATION:
- Ranks: 2,3,4,5,6,7,8,9,T (ten),J,Q,K,A
- Suits: s (spades ♠), h (hearts ♥), d (diamonds ♦), c (clubs ♣)
- Examples: As=Ace of spades, Td=Ten of diamonds, 7h=Seven of hearts

Return ONLY the JSON object, nothing else."""

    def __init__(self, 
                 model_path: Optional[str] = None,
                 device: str = 'auto',
                 use_quantization: bool = True,
                 api_fallback: bool = True):
        """
        Initialize the multi-modal vision detector.
        
        Args:
            model_path: Path to fine-tuned QWEN model (or Hugging Face model ID)
            device: Device to run model ('auto', 'cuda', 'cpu')
            use_quantization: Use 4-bit quantization for memory efficiency
            api_fallback: Fall back to OpenAI API if local model unavailable
        """
        self.model_path = model_path
        self.use_quantization = use_quantization
        self.api_fallback = api_fallback
        
        # Determine device
        if device == 'auto':
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Model components (loaded lazily)
        self.model = None
        self.processor = None
        self._model_loaded = False
        
        # API fallback client
        self._api_client = None
        
        print(f"[MultiModalVision] Initialized with device={self.device}")
    
    def _load_model(self):
        """Lazy load the QWEN model."""
        if self._model_loaded:
            return
        
        if not self.model_path:
            print("[MultiModalVision] No model path provided, using API fallback")
            self._model_loaded = True
            return
        
        if not TORCH_AVAILABLE:
            print("[MultiModalVision] PyTorch not available, using API fallback")
            self._model_loaded = True
            return
        
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            print(f"[MultiModalVision] Loading model from {self.model_path}...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model with optional quantization
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16 if self.device.type == 'cuda' else torch.float32,
            }
            
            if self.use_quantization and self.device.type == 'cuda':
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                except ImportError:
                    print("[MultiModalVision] bitsandbytes not available, skipping quantization")
            
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            if not self.use_quantization:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self._model_loaded = True
            print(f"[MultiModalVision] Model loaded successfully")
            
        except Exception as e:
            print(f"[MultiModalVision] Failed to load model: {e}")
            print("[MultiModalVision] Falling back to API mode")
            self._model_loaded = True
    
    def _get_api_client(self):
        """Get or create API client for fallback."""
        if self._api_client is None:
            try:
                from openai import OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    self._api_client = OpenAI(api_key=api_key)
            except ImportError:
                pass
        return self._api_client
    
    def detect(self, screenshot_path: str) -> DetectedGameState:
        """
        Detect game state from screenshot.
        
        Args:
            screenshot_path: Path to screenshot image file
            
        Returns:
            DetectedGameState with all detected information
        """
        # Ensure model is loaded
        self._load_model()
        
        # Try local model first
        if self.model is not None and self.processor is not None:
            return self._detect_local(screenshot_path)
        
        # Fall back to API
        if self.api_fallback:
            return self._detect_api(screenshot_path)
        
        # Return mock state if no detection method available
        return self._get_mock_state()
    
    def _detect_local(self, screenshot_path: str) -> DetectedGameState:
        """Detect using local QWEN model."""
        if not PIL_AVAILABLE:
            return self._get_mock_state()
        
        try:
            # Load image
            image = Image.open(screenshot_path).convert('RGB')
            
            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.DETECTION_PROMPT}
                    ]
                }
            ]
            
            # Process input
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # Decode response
            response = self.processor.decode(
                output_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Parse and validate response
            return self._parse_response(response)
            
        except Exception as e:
            print(f"[MultiModalVision] Local detection failed: {e}")
            return self._get_mock_state()
    
    def _detect_api(self, screenshot_path: str) -> DetectedGameState:
        """Detect using OpenAI API as fallback."""
        client = self._get_api_client()
        if client is None:
            print("[MultiModalVision] API client not available")
            return self._get_mock_state()
        
        try:
            import base64
            
            # Read and encode image
            with open(screenshot_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Call API
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.DETECTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                            }
                        ]
                    }
                ],
                max_tokens=512,
                temperature=0
            )
            
            content = response.choices[0].message.content
            return self._parse_response(content)
            
        except Exception as e:
            print(f"[MultiModalVision] API detection failed: {e}")
            return self._get_mock_state()
    
    def _parse_response(self, response: str) -> DetectedGameState:
        """Parse model response into DetectedGameState."""
        try:
            # Clean response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```"):
                lines = response.split('\n')
                response = '\n'.join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            # Find JSON object in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                state = DetectedGameState.from_dict(data)
                state.raw_response = response
                
                # Validate detected state
                state = self._validate_state(state)
                return state
            
            print(f"[MultiModalVision] No JSON found in response: {response[:100]}...")
            return self._get_mock_state()
            
        except json.JSONDecodeError as e:
            print(f"[MultiModalVision] JSON parse error: {e}")
            return self._get_mock_state()
        except Exception as e:
            print(f"[MultiModalVision] Parse error: {e}")
            return self._get_mock_state()
    
    def _validate_state(self, state: DetectedGameState) -> DetectedGameState:
        """Validate and clean detected game state."""
        # Check for duplicate cards
        all_cards = [c.to_string() for c in state.hole_cards + state.community_cards]
        if len(all_cards) != len(set(all_cards)):
            print("[MultiModalVision] Warning: Duplicate cards detected")
            state.confidence *= 0.5
        
        # Validate card counts
        if len(state.hole_cards) != 2:
            print(f"[MultiModalVision] Warning: Expected 2 hole cards, got {len(state.hole_cards)}")
            state.confidence *= 0.7
        
        # Validate street vs community cards
        expected_community = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}
        expected = expected_community.get(state.street, 0)
        if len(state.community_cards) != expected:
            print(f"[MultiModalVision] Warning: Street={state.street} expects {expected} cards, got {len(state.community_cards)}")
            # Auto-correct street if possible
            actual_count = len(state.community_cards)
            for street, count in expected_community.items():
                if count == actual_count:
                    state.street = street
                    break
        
        # Validate pot size
        state.pot_size = max(0, state.pot_size)
        state.current_bet = max(0, state.current_bet)
        state.player_stack = max(0, state.player_stack)
        state.opponent_stack = max(0, state.opponent_stack)
        
        return state
    
    def _get_mock_state(self) -> DetectedGameState:
        """Return mock state for testing when detection unavailable."""
        return DetectedGameState(
            hole_cards=[DetectedCard('A', 's'), DetectedCard('K', 's')],
            community_cards=[],
            pot_size=30,
            current_bet=20,
            player_stack=1000,
            opponent_stack=1000,
            street='preflop',
            action_required=True,
            available_actions=['fold', 'call', 'raise'],
            confidence=0.0,
            raw_response="MOCK: No detection method available"
        )
    
    def detect_batch(self, screenshot_paths: List[str]) -> List[DetectedGameState]:
        """
        Detect game states from multiple screenshots.
        
        Args:
            screenshot_paths: List of screenshot paths
            
        Returns:
            List of DetectedGameState objects
        """
        return [self.detect(path) for path in screenshot_paths]


# Convenience function for quick detection
def detect_game_state(screenshot_path: str, 
                      model_path: Optional[str] = None) -> DetectedGameState:
    """
    Quick detection of poker game state from screenshot.
    
    Args:
        screenshot_path: Path to screenshot
        model_path: Optional path to fine-tuned model
        
    Returns:
        DetectedGameState with detected information
    """
    detector = MultiModalVisionDetector(model_path=model_path)
    return detector.detect(screenshot_path)


__all__ = [
    'MultiModalVisionDetector',
    'DetectedGameState',
    'DetectedCard',
    'detect_game_state'
]
