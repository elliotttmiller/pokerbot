"""
LookaheadSolverAgent: Champion agent using continual re-solving via LookaheadSolver.
Integrates pluggable bucketing/action abstraction and logs validation metrics/model selection.
"""
from deepstack.core.lookahead_solver import LookaheadSolver
from .base_agent import BaseAgent

class LookaheadSolverAgent(BaseAgent):
    def __init__(self, config, name="LookaheadSolverAgent"):
        super().__init__(name)
        self.config = config
        self.solver = LookaheadSolver(config)
        self.training_mode = True
        self.validation_metrics = []
        self.best_model = None
        self.best_val_loss = float('inf')

    def choose_action(self, game_state):
        # Call continual re-solving at every decision
        action = self.solver.solve(game_state)
        return action

    def log_validation(self, val_loss, model):
        self.validation_metrics.append(val_loss)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model = model

    def get_best_model(self):
        return self.best_model
