"""
Action Executor - The "Hands" of the System.

Translates abstract decisions into physical GUI interactions.
Converts Action objects (e.g., "bet 500") into concrete mouse/keyboard
operations (e.g., move_to(x,y), click(), type('500')).

Key responsibilities:
- Map action types to screen coordinates
- Execute mouse movements and clicks
- Type bet amounts in bet boxes
- Handle timing and delays for reliability

Technology: PyAutoGUI or similar automation library.
"""

import time
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

from ..core.data_models import Action, ActionType


@dataclass
class ScreenCoordinates:
    """Screen coordinates for UI elements."""
    fold_button: Tuple[int, int] = (0, 0)
    check_button: Tuple[int, int] = (0, 0)
    call_button: Tuple[int, int] = (0, 0)
    raise_button: Tuple[int, int] = (0, 0)
    bet_input: Tuple[int, int] = (0, 0)
    all_in_button: Tuple[int, int] = (0, 0)
    confirm_button: Tuple[int, int] = (0, 0)


class ActionExecutor:
    """
    Translates poker decisions into GUI interactions.
    
    The "Hands" of the system - takes abstract Action objects and
    converts them into mouse/keyboard operations on the poker client.
    
    Usage:
        executor = ActionExecutor(coordinates)
        executor.execute(action)
    
    Configuration:
        Coordinates can be set via config file or calibration.
    """
    
    def __init__(self,
                 coordinates: Optional[ScreenCoordinates] = None,
                 action_delay: float = 0.5,
                 typing_delay: float = 0.05,
                 click_delay: float = 0.1,
                 simulation_mode: bool = True):
        """
        Initialize action executor.
        
        Args:
            coordinates: Screen coordinates for UI elements
            action_delay: Delay between major actions (seconds)
            typing_delay: Delay between keystrokes (seconds)
            click_delay: Delay after clicks (seconds)
            simulation_mode: If True, log actions without executing
        """
        self.coordinates = coordinates or ScreenCoordinates()
        self.action_delay = action_delay
        self.typing_delay = typing_delay
        self.click_delay = click_delay
        self.simulation_mode = simulation_mode
        
        # PyAutoGUI settings
        if PYAUTOGUI_AVAILABLE:
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.1
        
        print(f"[ActionExecutor] Initialized (simulation={simulation_mode})")
    
    def execute(self, action: Action) -> bool:
        """
        Execute the given action on the poker client.
        
        Args:
            action: Action to execute
            
        Returns:
            True if action was executed successfully
        """
        print(f"[ActionExecutor] Executing: {action}")
        
        if self.simulation_mode:
            return self._simulate_action(action)
        
        if not PYAUTOGUI_AVAILABLE:
            print("[ActionExecutor] PyAutoGUI not available")
            return False
        
        try:
            if action.action_type == ActionType.FOLD:
                return self._click_fold()
            elif action.action_type == ActionType.CHECK:
                return self._click_check()
            elif action.action_type == ActionType.CALL:
                return self._click_call()
            elif action.action_type == ActionType.BET:
                return self._enter_bet(action.amount)
            elif action.action_type == ActionType.RAISE:
                return self._enter_raise(action.amount)
            elif action.action_type == ActionType.ALL_IN:
                return self._click_all_in()
            else:
                print(f"[ActionExecutor] Unknown action type: {action.action_type}")
                return False
                
        except Exception as e:
            print(f"[ActionExecutor] Error executing action: {e}")
            return False
    
    def _simulate_action(self, action: Action) -> bool:
        """Simulate action without executing (for testing)."""
        action_str = action.to_command()
        print(f"[ActionExecutor] SIMULATION: Would execute '{action_str}'")
        time.sleep(0.1)  # Small delay for realism
        return True
    
    def _click_fold(self) -> bool:
        """Click the fold button."""
        return self._click_button(self.coordinates.fold_button, "FOLD")
    
    def _click_check(self) -> bool:
        """Click the check button."""
        return self._click_button(self.coordinates.check_button, "CHECK")
    
    def _click_call(self) -> bool:
        """Click the call button."""
        return self._click_button(self.coordinates.call_button, "CALL")
    
    def _click_all_in(self) -> bool:
        """Click the all-in button."""
        return self._click_button(self.coordinates.all_in_button, "ALL-IN")
    
    def _enter_bet(self, amount: int) -> bool:
        """Enter a bet amount."""
        # Click bet input field
        if not self._click_button(self.coordinates.bet_input, "BET INPUT"):
            return False
        
        time.sleep(self.click_delay)
        
        # Clear existing text and type amount
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(0.05)
        pyautogui.typewrite(str(amount), interval=self.typing_delay)
        
        time.sleep(self.click_delay)
        
        # Click confirm/raise button
        return self._click_button(self.coordinates.raise_button, "CONFIRM BET")
    
    def _enter_raise(self, amount: int) -> bool:
        """Enter a raise amount."""
        # Same as bet in most UIs
        return self._enter_bet(amount)
    
    def _click_button(self, coords: Tuple[int, int], button_name: str) -> bool:
        """Click a button at given coordinates."""
        x, y = coords
        
        if x == 0 and y == 0:
            print(f"[ActionExecutor] No coordinates set for {button_name}")
            return False
        
        print(f"[ActionExecutor] Clicking {button_name} at ({x}, {y})")
        
        try:
            pyautogui.moveTo(x, y, duration=0.2)
            time.sleep(self.click_delay)
            pyautogui.click()
            time.sleep(self.action_delay)
            return True
        except Exception as e:
            print(f"[ActionExecutor] Click failed: {e}")
            return False
    
    def calibrate(self, element: str) -> Tuple[int, int]:
        """
        Calibrate screen position for an element.
        
        Args:
            element: Element name ('fold', 'call', 'bet_input', etc.)
            
        Returns:
            Tuple of (x, y) coordinates
        """
        print(f"[ActionExecutor] Calibrating '{element}'...")
        print("Move mouse to element and press Enter...")
        
        if not PYAUTOGUI_AVAILABLE:
            return (0, 0)
        
        try:
            input()  # Wait for user to position mouse
            x, y = pyautogui.position()
            print(f"[ActionExecutor] Captured: ({x}, {y})")
            
            # Update coordinates
            element_lower = element.lower()
            if element_lower == 'fold':
                self.coordinates.fold_button = (x, y)
            elif element_lower == 'check':
                self.coordinates.check_button = (x, y)
            elif element_lower == 'call':
                self.coordinates.call_button = (x, y)
            elif element_lower in ['raise', 'bet']:
                self.coordinates.raise_button = (x, y)
            elif element_lower == 'bet_input':
                self.coordinates.bet_input = (x, y)
            elif element_lower == 'all_in':
                self.coordinates.all_in_button = (x, y)
            elif element_lower == 'confirm':
                self.coordinates.confirm_button = (x, y)
            
            return (x, y)
            
        except Exception as e:
            print(f"[ActionExecutor] Calibration error: {e}")
            return (0, 0)
    
    def set_coordinates(self, coords_dict: Dict[str, Tuple[int, int]]):
        """
        Set coordinates from dictionary.
        
        Args:
            coords_dict: Dictionary mapping element names to (x, y) tuples
        """
        if 'fold' in coords_dict:
            self.coordinates.fold_button = coords_dict['fold']
        if 'check' in coords_dict:
            self.coordinates.check_button = coords_dict['check']
        if 'call' in coords_dict:
            self.coordinates.call_button = coords_dict['call']
        if 'raise' in coords_dict:
            self.coordinates.raise_button = coords_dict['raise']
        if 'bet_input' in coords_dict:
            self.coordinates.bet_input = coords_dict['bet_input']
        if 'all_in' in coords_dict:
            self.coordinates.all_in_button = coords_dict['all_in']
        if 'confirm' in coords_dict:
            self.coordinates.confirm_button = coords_dict['confirm']
    
    def load_coordinates_from_config(self, config: Dict):
        """Load coordinates from configuration dictionary."""
        coords = config.get('screen_coordinates', {})
        self.set_coordinates(coords)
    
    def get_coordinates_dict(self) -> Dict[str, Tuple[int, int]]:
        """Get current coordinates as dictionary."""
        return {
            'fold': self.coordinates.fold_button,
            'check': self.coordinates.check_button,
            'call': self.coordinates.call_button,
            'raise': self.coordinates.raise_button,
            'bet_input': self.coordinates.bet_input,
            'all_in': self.coordinates.all_in_button,
            'confirm': self.coordinates.confirm_button
        }
    
    def set_simulation_mode(self, simulation: bool):
        """Enable or disable simulation mode."""
        self.simulation_mode = simulation
        print(f"[ActionExecutor] Simulation mode: {simulation}")


__all__ = ['ActionExecutor', 'ScreenCoordinates']
