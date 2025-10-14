# Quick Start Guide

This guide will help you get started with the Poker Bot quickly.

## Installation

1. Install Python 3.8 or higher
2. Clone the repository and navigate to it
3. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Tests

Verify the installation by running the test suite:

```bash
python test.py
```

You should see all tests passing.

## Quick Example

Run the quick example to see the poker bot in action:

```bash
python example.py
```

This will:
1. Train a DQN agent for 100 episodes
2. Evaluate all agents (DQN, Fixed Strategy, Random)
3. Run a head-to-head comparison

## Training Your Own Agent

Train a DQN agent for 1000 episodes:

```bash
python train.py --episodes 1000 --verbose
```

The trained model will be saved in the `models/` directory.

## Evaluating Agents

Compare all three agents over 1000 hands:

```bash
python evaluate.py --num-hands 1000 --agents dqn fixed random --verbose
```

To evaluate with a trained model:

```bash
python evaluate.py --num-hands 1000 --model-path models/model_final.h5 --verbose
```

## Playing on an Online Poker Site

**Note**: This requires an OpenAI API key for vision-based game detection.

1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

2. Open your poker website (e.g., www.247freepoker.com)

3. Run the bot:
   ```bash
   python play.py --agent fixed --max-hands 20 --delay 2.0 --verbose
   ```

The bot will:
- Capture screenshots of the poker table
- Detect the game state using GPT-4 Vision
- Make decisions using the chosen agent
- Execute actions on screen

## Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` to set your preferences:
- `OPENAI_API_KEY`: Your OpenAI API key
- `STARTING_STACK`: Initial chip stack (default: 1000)
- `SMALL_BLIND`: Small blind amount (default: 10)
- `BIG_BLIND`: Big blind amount (default: 20)
- Other training and evaluation parameters

## Tips

1. **Training**: Start with fewer episodes (100-500) for quick testing, then increase for better performance
2. **Evaluation**: Run at least 1000 hands for statistically significant results
3. **Playing**: Use `--delay` parameter to control action speed (2-3 seconds recommended)
4. **Fixed Strategy**: Generally performs better than untrained DQN and random agents
5. **DQN Training**: Requires significant training (10,000+ episodes) to outperform fixed strategy

## Common Issues

### Import Errors
If you get module import errors, make sure you've activated the virtual environment and installed all dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### TensorFlow Warnings
You may see TensorFlow warnings about CPU optimizations. These can be safely ignored for testing purposes.

### Vision Detection Issues
If vision detection isn't working:
1. Ensure your OpenAI API key is set correctly
2. Check that the screenshot captures the poker table correctly
3. Try with mock data first (bot will use fallback data without API key)

## Next Steps

1. Read the full README.md for detailed information
2. Explore the code in the `src/` directory
3. Experiment with different agent configurations
4. Try training a DQN agent with different hyperparameters
5. Customize the fixed strategy in `src/agents/fixed_strategy_agent.py`

## Support

For issues or questions:
1. Check the README.md for detailed documentation
2. Review the example.py for usage patterns
3. Examine the test.py for API examples

Happy playing! 🎰♠️♥️♦️♣️
