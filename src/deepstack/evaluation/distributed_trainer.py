"""Distributed training with multiprocessing for faster CFR convergence."""

import multiprocessing as mp
from multiprocessing import Manager, Pool
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from src.agents.advanced_cfr import AdvancedCFRAgent
from src.agents.cfr_agent import CFRAgent
from ..game import GameState


class DistributedTrainer:
    """
    Parallel CFR training using multiprocessing.
    
    Speeds up training by distributing iterations across multiple CPU cores.
    """
    
    def __init__(self,
                 agent: CFRAgent,
                 n_workers: int = None,
                 use_manager: bool = True):
        """
        Initialize distributed trainer.
        
        Args:
            agent: CFR agent to train
            n_workers: Number of worker processes (defaults to CPU count)
            use_manager: Use multiprocessing Manager for shared state
        """
        self.agent = agent
        self.n_workers = n_workers or mp.cpu_count()
        self.use_manager = use_manager
        
        if use_manager:
            self.manager = Manager()
            self.shared_regret = self.manager.dict()
            self.shared_strategy = self.manager.dict()
        else:
            self.shared_regret = {}
            self.shared_strategy = {}
    
    def train_parallel(self,
                      num_iterations: int = 10000,
                      save_interval: int = 1000,
                      save_dir: str = "models",
                      verbose: bool = True):
        """
        Train agent in parallel across multiple processes.
        
        Args:
            num_iterations: Total training iterations
            save_interval: Save agent every N iterations
            save_dir: Directory to save checkpoints
            verbose: Print progress
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Distribute iterations across workers
        iterations_per_worker = num_iterations // self.n_workers
        
        if verbose:
            print(f"Starting distributed training with {self.n_workers} workers")
            print(f"Total iterations: {num_iterations}")
            print(f"Iterations per worker: {iterations_per_worker}")
        
        # Create work batches
        work_batches = [
            (worker_id, iterations_per_worker)
            for worker_id in range(self.n_workers)
        ]
        
        # Run parallel training
        with Pool(processes=self.n_workers) as pool:
            results = pool.starmap(
                self._train_worker,
                work_batches
            )
        
        # Merge results from all workers
        self._merge_worker_results(results)
        
        if verbose:
            print(f"Training complete. Total infosets: {len(self.agent.infosets)}")
        
        # Save final model
        save_path = os.path.join(save_dir, "distributed_final.cfr")
        self.agent.save_strategy(save_path)
        if verbose:
            print(f"Model saved to {save_path}")
    
    def _train_worker(self, worker_id: int, num_iterations: int) -> Dict:
        """
        Worker function for parallel training.
        
        Args:
            worker_id: Unique worker identifier
            num_iterations: Iterations for this worker
        
        Returns:
            Dictionary with worker results
        """
        # Create worker-specific agent
        worker_agent = AdvancedCFRAgent(name=f"Worker{worker_id}")
        
        # Train for assigned iterations
        for t in range(num_iterations):
            game_state = GameState(num_players=2)
            game_state.reset()
            
            for player_idx in range(2):
                game_state.reset()
                worker_agent._cfr_iteration(game_state, player_idx, 1.0, 1.0)
        
        # Return worker results
        return {
            'worker_id': worker_id,
            'infosets': dict(worker_agent.infosets),
            'iterations': num_iterations
        }
    
    def _merge_worker_results(self, results: list):
        """
        Merge results from all workers into main agent.
        
        Args:
            results: List of result dictionaries from workers
        """
        # Merge infosets from all workers
        for result in results:
            worker_infosets = result['infosets']
            
            for infoset_key, worker_infoset in worker_infosets.items():
                if infoset_key in self.agent.infosets:
                    # Merge regrets and strategies
                    main_infoset = self.agent.infosets[infoset_key]
                    main_infoset.regret_sum += worker_infoset.regret_sum
                    main_infoset.strategy_sum += worker_infoset.strategy_sum
                else:
                    # Add new infoset
                    self.agent.infosets[infoset_key] = worker_infoset
            
            self.agent.iterations += result['iterations']


class AsyncDistributedTrainer(DistributedTrainer):
    """
    Asynchronous distributed trainer with periodic synchronization.
    
    Workers train independently and synchronize periodically for better speed.
    """
    
    def __init__(self,
                 agent: CFRAgent,
                 n_workers: int = None,
                 sync_interval: int = 100):
        """
        Initialize async distributed trainer.
        
        Args:
            agent: CFR agent to train
            n_workers: Number of worker processes
            sync_interval: Synchronize workers every N iterations
        """
        super().__init__(agent, n_workers, use_manager=True)
        self.sync_interval = sync_interval
    
    def train_async(self,
                   num_iterations: int = 10000,
                   save_interval: int = 1000,
                   save_dir: str = "models",
                   verbose: bool = True):
        """
        Train agent asynchronously with periodic sync.
        
        Args:
            num_iterations: Total training iterations
            save_interval: Save agent every N iterations
            save_dir: Directory to save checkpoints
            verbose: Print progress
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if verbose:
            print(f"Starting async distributed training with {self.n_workers} workers")
            print(f"Synchronization interval: {self.sync_interval}")
        
        # Calculate sync points
        n_syncs = num_iterations // self.sync_interval
        
        for sync in range(n_syncs):
            # Train workers until next sync
            iterations_per_worker = self.sync_interval // self.n_workers
            
            work_batches = [
                (worker_id, iterations_per_worker, self.shared_regret, self.shared_strategy)
                for worker_id in range(self.n_workers)
            ]
            
            with Pool(processes=self.n_workers) as pool:
                results = pool.starmap(
                    self._train_worker_async,
                    work_batches
                )
            
            # Synchronize
            self._sync_workers(results)
            
            if verbose and (sync + 1) % 10 == 0:
                print(f"Sync {sync + 1}/{n_syncs} complete")
            
            # Save periodically
            if (sync + 1) % (save_interval // self.sync_interval) == 0:
                save_path = os.path.join(save_dir, f"async_checkpoint_{sync + 1}.cfr")
                self.agent.save_strategy(save_path)
        
        # Save final model
        save_path = os.path.join(save_dir, "async_final.cfr")
        self.agent.save_strategy(save_path)
        if verbose:
            print(f"Async training complete. Model saved to {save_path}")
    
    def _train_worker_async(self,
                           worker_id: int,
                           num_iterations: int,
                           shared_regret: Dict,
                           shared_strategy: Dict) -> Dict:
        """
        Async worker that reads from and writes to shared memory.
        
        Args:
            worker_id: Worker identifier
            num_iterations: Iterations to run
            shared_regret: Shared regret dictionary
            shared_strategy: Shared strategy dictionary
        
        Returns:
            Worker results
        """
        worker_agent = AdvancedCFRAgent(name=f"AsyncWorker{worker_id}")
        
        # Load initial state from shared memory
        for key, value in shared_regret.items():
            if key in worker_agent.infosets:
                worker_agent.infosets[key].regret_sum = value
        
        # Train
        for t in range(num_iterations):
            game_state = GameState(num_players=2)
            game_state.reset()
            
            for player_idx in range(2):
                game_state.reset()
                worker_agent._cfr_iteration(game_state, player_idx, 1.0, 1.0)
        
        return {
            'worker_id': worker_id,
            'infosets': dict(worker_agent.infosets),
            'iterations': num_iterations
        }
    
    def _sync_workers(self, results: list):
        """
        Synchronize worker results into shared memory.
        
        Args:
            results: Worker results to sync
        """
        self._merge_worker_results(results)
        
        # Update shared memory
        for infoset_key, infoset in self.agent.infosets.items():
            self.shared_regret[infoset_key] = infoset.regret_sum
            self.shared_strategy[infoset_key] = infoset.strategy_sum
