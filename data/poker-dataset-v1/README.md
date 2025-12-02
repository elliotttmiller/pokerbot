# Poker Dataset v1

This directory contains training data for the VLM perception model.

## Expected Structure
```
poker-dataset-v1/
├── images/           # Poker table screenshots
│   ├── 001.png
│   ├── 002.png
│   └── ...
├── annotations.json  # Ground truth game state labels
└── splits/           # Train/val/test splits
    ├── train.json
    ├── val.json
    └── test.json
```

## Annotation Format
Each annotation contains:
```json
{
    "image_id": "001.png",
    "hole_cards": ["As", "Ks"],
    "community_cards": ["Qh", "Jd", "Tc"],
    "pot_size": 150,
    "current_bet": 50,
    "player_stack": 1000,
    "opponent_stack": 1200,
    "street": "flop",
    "action_required": true,
    "available_actions": ["fold", "call", "raise"]
}
```

## Data Collection
See `docs/DATA_COLLECTION.md` for guidelines on collecting training data.
