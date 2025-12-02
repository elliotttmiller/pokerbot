"""
VLM Service - The "Croupier" Perception Engine.

Converts unstructured visual data (pixels) into structured game information (data).
Uses a fine-tuned Vision-Language Model (QWEN 2.5-VL) for game state extraction.

Key responsibilities:
- Load and manage VLM model (single VRAM initialization)
- Process poker table screenshots
- Extract structured GameState from visual input
- Validate perception results

References:
- DeepStack: Game state representation
- QWEN 2.5-VL: Vision-language model architecture
"""

import json
import os
import re
from typing import Optional, Dict, Any
from pathlib import Path

# Optional imports
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..core.data_models import GameState, Card, PlayerState, BettingRound, ActionType


class VLMService:
    """
    Vision-Language Model Service for game state perception.
    
    The "Croupier" - its only job is to achieve flawless perception.
    It looks at the screen and produces a perfect, structured 
    representation of the game state. It knows nothing of strategy.
    
    Usage:
        service = VLMService(model_path="models/perception_v1")
        game_state = service.get_game_state(screenshot_image)
    """
    
    # Detection prompt optimized for poker game state extraction
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
5. Read player and opponent stack sizes

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
- Examples: As=Ace of spades, Td=Ten of diamonds, 7h=Seven of hearts

Return ONLY the JSON object, nothing else."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'auto',
                 use_quantization: bool = True,
                 api_fallback: bool = True):
        """
        Initialize VLM Service.
        
        Args:
            model_path: Path to fine-tuned QWEN model
            device: Computation device ('auto', 'cuda', 'cpu')
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
            self.device = torch.device(device) if TORCH_AVAILABLE else 'cpu'
        
        # Model components (loaded lazily to conserve memory)
        self.model = None
        self.processor = None
        self._model_loaded = False
        
        # API client for fallback
        self._api_client = None
        
        print(f"[VLMService] Initialized (device={self.device})")
    
    def _load_model(self):
        """Lazy load the VLM model into VRAM."""
        if self._model_loaded:
            return
        
        if not self.model_path or not os.path.exists(self.model_path):
            print("[VLMService] No model path provided, using API fallback")
            self._model_loaded = True
            return
        
        if not TORCH_AVAILABLE:
            print("[VLMService] PyTorch not available, using API fallback")
            self._model_loaded = True
            return
        
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            
            print(f"[VLMService] Loading model from {self.model_path}...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Model loading configuration
            model_kwargs = {
                'trust_remote_code': True,
                'torch_dtype': torch.float16 if self.device.type == 'cuda' else torch.float32,
            }
            
            # Add quantization if requested and available
            if self.use_quantization and self.device.type == 'cuda':
                try:
                    from transformers import BitsAndBytesConfig
                    model_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                except ImportError:
                    print("[VLMService] bitsandbytes not available, skipping quantization")
            
            # Load model
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            
            if not self.use_quantization:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self._model_loaded = True
            print(f"[VLMService] Model loaded successfully")
            
        except Exception as e:
            print(f"[VLMService] Failed to load model: {e}")
            self._model_loaded = True
    
    def _get_api_client(self):
        """Get or create OpenAI API client for fallback."""
        if self._api_client is None:
            try:
                from openai import OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    self._api_client = OpenAI(api_key=api_key)
            except ImportError:
                pass
        return self._api_client
    
    def get_game_state(self, image) -> GameState:
        """
        Extract game state from screenshot image.
        
        Primary method for the perception pipeline.
        
        Args:
            image: PIL Image of the poker table
            
        Returns:
            Validated GameState object
        """
        # Ensure model is loaded
        self._load_model()
        
        # Try local model first
        if self.model is not None and self.processor is not None:
            return self._detect_local(image)
        
        # Try API fallback
        if self.api_fallback:
            return self._detect_api(image)
        
        # Return mock state if no detection available
        return self._get_mock_state()
    
    def _detect_local(self, image) -> GameState:
        """Run detection using local VLM model."""
        if not PIL_AVAILABLE:
            return self._get_mock_state()
        
        try:
            # Ensure image is PIL Image
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            
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
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # Decode response
            response = self.processor.decode(
                output_ids[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Parse response
            return self._parse_response(response)
            
        except Exception as e:
            print(f"[VLMService] Local detection failed: {e}")
            return self._get_mock_state()
    
    def _detect_api(self, image) -> GameState:
        """Run detection using OpenAI API."""
        client = self._get_api_client()
        if client is None:
            print("[VLMService] API client not available")
            return self._get_mock_state()
        
        try:
            import base64
            from io import BytesIO
            
            # Convert image to base64
            if isinstance(image, str):
                with open(image, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")
            else:
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
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
                                "image_url": {"url": f"data:image/png;base64,{image_data}"}
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
            print(f"[VLMService] API detection failed: {e}")
            return self._get_mock_state()
    
    def _parse_response(self, response: str) -> GameState:
        """Parse VLM response into GameState."""
        try:
            # Clean response
            response = response.strip()
            
            # Remove markdown code blocks
            if response.startswith("```"):
                lines = response.split('\n')
                response = '\n'.join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            
            # Extract JSON
            json_str = self._extract_json(response)
            if not json_str:
                print(f"[VLMService] No JSON found in response")
                return self._get_mock_state()
            
            data = json.loads(json_str)
            
            # Convert to GameState
            return self._dict_to_game_state(data, response)
            
        except json.JSONDecodeError as e:
            print(f"[VLMService] JSON parse error: {e}")
            return self._get_mock_state()
        except Exception as e:
            print(f"[VLMService] Parse error: {e}")
            return self._get_mock_state()
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text using bracket matching."""
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        depth = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text[start_idx:], start=start_idx):
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
                    return text[start_idx:i+1]
        
        return None
    
    def _dict_to_game_state(self, data: Dict, raw_response: str) -> GameState:
        """Convert parsed dictionary to GameState."""
        # Parse hole cards
        hole_cards = []
        for card_str in data.get('hole_cards', []):
            try:
                hole_cards.append(Card.from_string(card_str))
            except ValueError:
                pass
        
        # Parse community cards
        community_cards = []
        for card_str in data.get('community_cards', []):
            try:
                community_cards.append(Card.from_string(card_str))
            except ValueError:
                pass
        
        # Determine street from community cards
        street_map = {0: BettingRound.PREFLOP, 3: BettingRound.FLOP, 
                      4: BettingRound.TURN, 5: BettingRound.RIVER}
        street = street_map.get(len(community_cards), BettingRound.PREFLOP)
        
        # Override with explicit street if valid
        explicit_street = data.get('street', '').lower()
        if explicit_street in ['preflop', 'flop', 'turn', 'river']:
            street = BettingRound(explicit_street)
        
        # Parse available actions
        available_actions = []
        for action_str in data.get('available_actions', ['fold', 'call', 'raise']):
            try:
                available_actions.append(ActionType(action_str.lower()))
            except ValueError:
                pass
        
        if not available_actions:
            available_actions = [ActionType.FOLD, ActionType.CALL, ActionType.RAISE]
        
        # Build player states
        player = PlayerState(
            stack=max(0, data.get('player_stack', 1000)),
            current_bet=max(0, data.get('current_bet', 0))
        )
        
        opponent = PlayerState(
            stack=max(0, data.get('opponent_stack', 1000))
        )
        
        # Calculate confidence
        confidence = 1.0
        if len(hole_cards) != 2:
            confidence *= 0.7
        if len(community_cards) not in [0, 3, 4, 5]:
            confidence *= 0.7
        
        # Build and validate GameState
        try:
            state = GameState(
                hole_cards=hole_cards if len(hole_cards) == 2 else [],
                community_cards=community_cards if len(community_cards) in [0, 3, 4, 5] else [],
                pot_size=max(0, data.get('pot_size', 0)),
                current_bet=max(0, data.get('current_bet', 0)),
                player=player,
                opponent=opponent,
                street=street,
                action_required=data.get('action_required', True),
                available_actions=available_actions,
                confidence=confidence,
                raw_perception=raw_response
            )
            return state
        except Exception as e:
            print(f"[VLMService] GameState validation error: {e}")
            return self._get_mock_state()
    
    def _get_mock_state(self) -> GameState:
        """Return mock state for testing."""
        return GameState(
            hole_cards=[Card.from_string('As'), Card.from_string('Ks')],
            community_cards=[],
            pot_size=30,
            current_bet=20,
            player=PlayerState(stack=1000, current_bet=0),
            opponent=PlayerState(stack=1000),
            street=BettingRound.PREFLOP,
            action_required=True,
            available_actions=[ActionType.FOLD, ActionType.CALL, ActionType.RAISE],
            confidence=0.0,
            raw_perception="MOCK: No detection method available"
        )


__all__ = ['VLMService']
