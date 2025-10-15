"""
ActionAbstraction: Pluggable action abstraction for DeepStack agents.
"""
class ActionAbstraction:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.strategy = config.get('action_abstraction_strategy', 'default')
        self.strategies = {
            'default': self.default_actions,
            'pot': self.pot_actions,
            # Add more strategies here
        }

    def get_actions(self, state):
        return self.strategies.get(self.strategy, self.default_actions)(state)

    def default_actions(self, state):
        # Placeholder: fold, call, raise
        return ['fold', 'call', 'raise']

    def pot_actions(self, state):
        # TODO: Implement pot-based abstraction
        # Placeholder: use default for now
        return self.default_actions(state)
