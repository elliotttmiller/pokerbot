"""Vision-based game state detection."""

import os
from typing import Dict, List, Optional

from ..game import Card


class VisionDetector:
    """Detects poker game state from screenshots using vision AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize vision detector.
        
        Args:
            api_key: OpenAI API key for GPT-4 Vision
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("Warning: OpenAI library not installed. Vision detection disabled.")
    
    def detect_game_state(self, screenshot_path: str) -> Dict:
        """
        Detect game state from screenshot.
        
        Args:
            screenshot_path: Path to screenshot image
        
        Returns:
            Dictionary with game state information:
            - is_game_over: bool
            - community_cards: List[str]
            - hole_cards: List[str]
            - pot_value: int
            - raised_amounts: List[int]
        """
        if not self.client:
            return self._mock_detection()
        
        import base64
        import json
        
        # Read and encode image
        with open(screenshot_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        
        # Create prompt
        prompt = self._create_detection_prompt()
        
        # Call GPT-4 Vision
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a poker game state detector. Extract game information from screenshots."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            content = self._clean_json(content)
            result = json.loads(content)
            
            return result
        
        except Exception as e:
            print(f"Error detecting game state: {e}")
            return self._mock_detection()
    
    def _create_detection_prompt(self) -> str:
        """Create prompt for vision detection."""
        return """
Look at this poker game screenshot and extract the following information:

1. Is the game over? (true/false)
2. Community cards (cards on the table visible to all)
3. Hole cards (your private cards)
4. Current pot value (total chips in the pot)
5. Who raised and how much

Return the information in this exact JSON format:
{
    "is_game_over": false,
    "community_cards": ["A-S", "K-H", "Q-D"],
    "hole_cards": ["J-C", "10-S"],
    "pot_value": 150,
    "raised_amounts": [20, 40]
}

Card format: Rank-Suit
- Ranks: 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K, A
- Suits: S (Spades), H (Hearts), D (Diamonds), C (Clubs)

Return ONLY the JSON, no other text.
"""
    
    def _clean_json(self, content: str) -> str:
        """Clean JSON response from markdown formatting."""
        content = content.strip()
        
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        
        if content.endswith("```"):
            content = content[:-3]
        
        return content.strip()
    
    def _mock_detection(self) -> Dict:
        """Return mock detection for testing."""
        return {
            "is_game_over": False,
            "community_cards": [],
            "hole_cards": ["A-S", "K-S"],
            "pot_value": 30,
            "raised_amounts": [20]
        }
    
    def parse_cards(self, card_strings: List[str]) -> List[Card]:
        """
        Parse card strings to Card objects.
        
        Args:
            card_strings: List of card strings like ["A-S", "K-H"]
        
        Returns:
            List of Card objects
        """
        cards = []
        for card_str in card_strings:
            try:
                card = Card.from_string(card_str)
                cards.append(card)
            except ValueError as e:
                print(f"Warning: Could not parse card '{card_str}': {e}")
        
        return cards
