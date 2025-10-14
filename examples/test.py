#!/usr/bin/env python3
"""Comprehensive test suite for poker bot."""

import sys


def test_imports():
    """Test all module imports."""
    print("Testing imports...")
    try:
        from src.game import Card, Deck, HandEvaluator, GameState, Action
        from src.agents import RandomAgent, FixedStrategyAgent
        from src.evaluation import Evaluator
        from src.utils import Config, Logger
        print("✓ All imports successful\n")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}\n")
        return False


def test_cards():
    """Test card system."""
    print("Testing card system...")
    try:
        from src.game import Card, Rank, Suit, Deck
        
        # Test card creation
        card1 = Card(Rank.ACE, Suit.SPADES)
        assert str(card1) == "A♠", "Card string representation failed"
        
        # Test card from string
        card2 = Card.from_string("K-H")
        assert card2.rank == Rank.KING, "Card parsing failed"
        assert card2.suit == Suit.HEARTS, "Card suit parsing failed"
        
        # Test deck
        deck = Deck()
        assert len(deck) == 52, "Deck should have 52 cards"
        
        dealt = deck.deal(5)
        assert len(dealt) == 5, "Should deal 5 cards"
        assert len(deck) == 47, "Deck should have 47 cards after dealing"
        
        print("✓ Card system tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Card system tests failed: {e}\n")
        return False


def test_hand_evaluation():
    """Test hand evaluation."""
    print("Testing hand evaluation...")
    try:
        from src.game import Card, Rank, Suit, HandEvaluator, HandRank
        
        # Test royal flush
        royal_flush = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.KING, Suit.SPADES),
            Card(Rank.QUEEN, Suit.SPADES),
            Card(Rank.JACK, Suit.SPADES),
            Card(Rank.TEN, Suit.SPADES)
        ]
        rank, _ = HandEvaluator.evaluate_hand(royal_flush)
        assert rank == HandRank.ROYAL_FLUSH, "Royal flush not detected"
        
        # Test pair
        pair = [
            Card(Rank.ACE, Suit.SPADES),
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.DIAMONDS),
            Card(Rank.QUEEN, Suit.CLUBS),
            Card(Rank.JACK, Suit.SPADES)
        ]
        rank, _ = HandEvaluator.evaluate_hand(pair)
        assert rank == HandRank.ONE_PAIR, "Pair not detected"
        
        # Test flush
        flush = [
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.HEARTS),
            Card(Rank.FIVE, Suit.HEARTS)
        ]
        rank, _ = HandEvaluator.evaluate_hand(flush)
        assert rank == HandRank.FLUSH, "Flush not detected"
        
        print("✓ Hand evaluation tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Hand evaluation tests failed: {e}\n")
        return False


def test_game_state():
    """Test game state management."""
    print("Testing game state...")
    try:
        from src.game import GameState, Action
        
        # Create game
        game = GameState(num_players=2, starting_stack=1000)
        game.reset()
        
        assert len(game.players) == 2, "Should have 2 players"
        assert game.pot == 30, "Pot should be 30 after blinds"
        assert len(game.players[0].hand) == 2, "Player should have 2 cards"
        assert len(game.community_cards) == 0, "Should have no community cards initially"
        
        # Test action
        success = game.apply_action(0, Action.CALL, 0)
        assert success, "Action should be successful"
        
        print("✓ Game state tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Game state tests failed: {e}\n")
        return False


def test_agents():
    """Test poker agents."""
    print("Testing agents...")
    try:
        from src.game import GameState
        from src.agents import RandomAgent, FixedStrategyAgent
        
        game = GameState(num_players=2)
        game.reset()
        
        player = game.players[0]
        
        # Test random agent
        random_agent = RandomAgent()
        action, raise_amount = random_agent.choose_action(
            hole_cards=player.hand,
            community_cards=game.community_cards,
            pot=game.pot,
            current_bet=game.current_bet,
            player_stack=player.stack,
            opponent_bet=game.current_bet
        )
        assert action is not None, "Random agent should return an action"
        
        # Test fixed strategy agent
        fixed_agent = FixedStrategyAgent()
        action, raise_amount = fixed_agent.choose_action(
            hole_cards=player.hand,
            community_cards=game.community_cards,
            pot=game.pot,
            current_bet=game.current_bet,
            player_stack=player.stack,
            opponent_bet=game.current_bet
        )
        assert action is not None, "Fixed agent should return an action"
        
        print("✓ Agent tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Agent tests failed: {e}\n")
        return False


def test_evaluator():
    """Test evaluation system."""
    print("Testing evaluator...")
    try:
        from src.agents import RandomAgent, FixedStrategyAgent
        from src.evaluation import Evaluator
        
        # Create agents
        random_agent = RandomAgent()
        fixed_agent = FixedStrategyAgent()
        
        # Create evaluator
        evaluator = Evaluator([random_agent, fixed_agent])
        
        # Run evaluation
        results = evaluator.evaluate_agents(num_hands=5, verbose=False)
        
        assert 'wins' in results, "Results should have wins"
        assert 'losses' in results, "Results should have losses"
        assert 'ties' in results, "Results should have ties"
        
        total_outcomes = (results['wins']['RandomAgent'] + 
                         results['wins']['FixedStrategy'] + 
                         results['ties'])
        assert total_outcomes == 5, "Should have 5 total outcomes"
        
        print("✓ Evaluator tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Evaluator tests failed: {e}\n")
        return False


def test_config():
    """Test configuration system."""
    print("Testing configuration...")
    try:
        from src.utils import Config
        
        config = Config()
        assert config.starting_stack == 1000, "Default starting stack should be 1000"
        assert config.small_blind == 10, "Default small blind should be 10"
        assert config.big_blind == 20, "Default big blind should be 20"
        
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict), "Config should convert to dict"
        
        print("✓ Configuration tests passed\n")
        return True
    except Exception as e:
        print(f"✗ Configuration tests failed: {e}\n")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("POKER BOT TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        test_imports,
        test_cards,
        test_hand_evaluation,
        test_game_state,
        test_agents,
        test_evaluator,
        test_config,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print("="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
