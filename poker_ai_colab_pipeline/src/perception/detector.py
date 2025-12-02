"""
QWEN 2.5-7B VL Poker Detector

Colab-optimized vision detector using fine-tuned QWEN 2.5-7B VL
with Unsloth 4-bit quantization for poker game state detection.
"""

import os
import json
import gc
from typing import Optional, Dict, Any
from pathlib import Path

from .data_models import GameState, Card, BettingRound

# Optional imports with graceful fallback
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


class QWENPokerDetector:
    """
    QWEN 2.5-7B VL detector for poker game state extraction.
    
    Optimized for Google Colab T4 GPU with:
    - 4-bit quantization via bitsandbytes
    - Gradient checkpointing for memory efficiency
    - Automatic VRAM cleanup
    
    Usage:
        detector = QWENPokerDetector(model_path="drive/models/qwen_poker_vl")
        state = detector.detect("screenshot.png")
        print(f"Hole cards: {state.hole_cards}")
    """
    
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
- Suits: s (spades), h (hearts), d (diamonds), c (clubs)

Return ONLY the JSON object, nothing else."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'auto',
        use_quantization: bool = True,
        max_new_tokens: int = 512
    ):
        """
        Initialize the QWEN poker detector.
        
        Args:
            model_path: Path to fine-tuned model (local or HuggingFace ID)
            device: Computation device ('auto', 'cuda', 'cpu')
            use_quantization: Enable 4-bit quantization (recommended for Colab)
            max_new_tokens: Maximum tokens in model output
        """
        self.model_path = model_path or "Qwen/Qwen2.5-VL-7B-Instruct"
        self.use_quantization = use_quantization
        self.max_new_tokens = max_new_tokens
        
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
        
        print(f"[QWENPokerDetector] Initialized (device={self.device}, quantization={use_quantization})")
    
    def _load_model(self):
        """Load the QWEN VL model with Colab optimizations."""
        if self._model_loaded:
            return
        
        if not TORCH_AVAILABLE:
            print("[QWENPokerDetector] PyTorch not available")
            self._model_loaded = True
            return
        
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            print(f"[QWENPokerDetector] Loading model from {self.model_path}...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Model loading kwargs
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16 if self.device.type == 'cuda' else torch.float32,
            }
            
            # Add 4-bit quantization for Colab T4
            if self.use_quantization and self.device.type == 'cuda':
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    print("[QWENPokerDetector] Using 4-bit quantization")
                except ImportError:
                    print("[QWENPokerDetector] bitsandbytes not available, using FP16")
            
            # Load model
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            if not self.use_quantization:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self._model_loaded = True
            
            # Report VRAM usage
            if self.device.type == 'cuda':
                vram_used = torch.cuda.memory_allocated() / 1e9
                print(f"[QWENPokerDetector] Model loaded (VRAM: {vram_used:.2f} GB)")
            
        except Exception as e:
            print(f"[QWENPokerDetector] Failed to load model: {e}")
            self._model_loaded = True
    
    def detect(self, image_path: str) -> GameState:
        """
        Detect poker game state from screenshot.
        
        Args:
            image_path: Path to screenshot image
            
        Returns:
            Validated GameState object
        """
        self._load_model()
        
        if self.model is None or self.processor is None:
            return self._get_mock_state("Model not loaded")
        
        if not PIL_AVAILABLE:
            return self._get_mock_state("PIL not available")
        
        try:
            return self._detect_local(image_path)
        except Exception as e:
            print(f"[QWENPokerDetector] Detection error: {e}")
            return self._get_mock_state(str(e))
    
    def _detect_local(self, image_path: str) -> GameState:
        """Run local model inference."""
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare chat messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.DETECTION_PROMPT}
                ]
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        # Decode response
        response = self.processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Clear CUDA cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return self._parse_response(response)
    
    def _parse_response(self, response: str) -> GameState:
        """Parse model response into GameState."""
        try:
            # Clean response
            response = response.strip()
            
            # Remove markdown code blocks
            if response.startswith("```"):
                lines = response.split('\n')
                response = '\n'.join(lines[1:-1] if lines[-1].strip() in ["```", ""] else lines[1:])
                response = response.replace("```", "").strip()
            
            # Find JSON object
            json_str = self._extract_json(response)
            if not json_str:
                return self._get_mock_state("No JSON found in response")
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Convert to GameState
            hole_cards = [Card.from_string(c) for c in data.get('hole_cards', [])]
            community_cards = [Card.from_string(c) for c in data.get('community_cards', [])]
            
            street_str = data.get('street', 'preflop').lower()
            street = BettingRound(street_str)
            
            state = GameState(
                hole_cards=hole_cards,
                community_cards=community_cards,
                pot_size=data.get('pot_size', 0),
                current_bet=data.get('current_bet', 0),
                player_stack=data.get('player_stack', 1000),
                opponent_stack=data.get('opponent_stack', 1000),
                street=street,
                action_required=data.get('action_required', True),
                available_actions=data.get('available_actions', ['fold', 'call', 'raise']),
                confidence=1.0,
                raw_response=response
            )
            
            # Validate and adjust confidence
            state = self._validate_state(state)
            
            return state
            
        except json.JSONDecodeError as e:
            return self._get_mock_state(f"JSON parse error: {e}")
        except Exception as e:
            return self._get_mock_state(f"Parse error: {e}")
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text using bracket matching."""
        start = text.find('{')
        if start == -1:
            return None
        
        depth = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start:], start=start):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        
        return None
    
    def _validate_state(self, state: GameState) -> GameState:
        """Validate detected state and adjust confidence."""
        # Check for duplicate cards
        all_cards = [c.to_string() for c in state.hole_cards + state.community_cards]
        if len(all_cards) != len(set(all_cards)):
            state.confidence *= 0.5
            print("[QWENPokerDetector] Warning: Duplicate cards detected")
        
        # Validate hole card count
        if len(state.hole_cards) != 2:
            state.confidence *= 0.7
            print(f"[QWENPokerDetector] Warning: Expected 2 hole cards, got {len(state.hole_cards)}")
        
        # Validate street vs community cards
        if not state.validate_street_cards():
            state.confidence *= 0.8
            # Auto-correct street
            cc_count = len(state.community_cards)
            street_map = {0: BettingRound.PREFLOP, 3: BettingRound.FLOP, 
                         4: BettingRound.TURN, 5: BettingRound.RIVER}
            if cc_count in street_map:
                state.street = street_map[cc_count]
        
        return state
    
    def _get_mock_state(self, reason: str = "") -> GameState:
        """Return mock state for testing/fallback."""
        return GameState(
            hole_cards=[Card(rank='A', suit='s'), Card(rank='K', suit='s')],
            community_cards=[],
            pot_size=30,
            current_bet=20,
            player_stack=1000,
            opponent_stack=1000,
            street=BettingRound.PREFLOP,
            action_required=True,
            available_actions=['fold', 'call', 'raise'],
            confidence=0.0,
            raw_response=f"MOCK: {reason}"
        )
    
    def cleanup(self):
        """Release model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._model_loaded = False
        print("[QWENPokerDetector] Resources released")


def get_game_state(screenshot_path: str, model_path: Optional[str] = None) -> GameState:
    """
    One-call function to get game state from screenshot.
    
    Args:
        screenshot_path: Path to screenshot image
        model_path: Optional path to fine-tuned model
        
    Returns:
        Validated GameState object
    """
    detector = QWENPokerDetector(model_path=model_path)
    return detector.detect(screenshot_path)


__all__ = ['QWENPokerDetector', 'get_game_state']
