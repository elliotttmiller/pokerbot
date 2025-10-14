"""Screen capture and control utilities."""

import platform
import subprocess
import time
from typing import Optional, Tuple


class ScreenController:
    """Controls screen capture and mouse/keyboard input."""
    
    def __init__(self):
        """Initialize screen controller."""
        self.system = platform.system()
        
        # Import PyAutoGUI if available
        try:
            import pyautogui
            self.pyautogui = pyautogui
            self.pyautogui.FAILSAFE = True
        except ImportError:
            print("Warning: PyAutoGUI not installed. Screen control disabled.")
            self.pyautogui = None
    
    def capture_screenshot(self, filepath: str):
        """
        Capture a screenshot and save to file.
        
        Args:
            filepath: Path to save screenshot
        """
        if self.system == "Darwin":  # macOS
            subprocess.run(["screencapture", "-C", filepath])
        elif self.pyautogui:
            screenshot = self.pyautogui.screenshot()
            screenshot.save(filepath)
        else:
            print("Warning: Screenshot not available on this platform")
    
    def get_screen_size(self) -> Tuple[int, int]:
        """
        Get screen dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        if self.pyautogui:
            return self.pyautogui.size()
        return (1920, 1080)  # Default
    
    def click_at_position(self, x: int, y: int, duration: float = 0.2):
        """
        Click at screen position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            duration: Duration of mouse movement
        """
        if not self.pyautogui:
            print(f"Would click at ({x}, {y})")
            return
        
        self.pyautogui.moveTo(x, y, duration=duration)
        time.sleep(0.1)
        self.pyautogui.click(x, y)
    
    def click_at_percentage(self, x_percent: float, y_percent: float, 
                           duration: float = 0.2):
        """
        Click at screen position specified as percentage.
        
        Args:
            x_percent: X position as percentage (0.0 to 1.0)
            y_percent: Y position as percentage (0.0 to 1.0)
            duration: Duration of mouse movement
        """
        screen_width, screen_height = self.get_screen_size()
        x = int(screen_width * x_percent)
        y = int(screen_height * y_percent)
        self.click_at_position(x, y, duration)
    
    def press_key(self, key: str, duration: float = 0.1):
        """
        Press a keyboard key.
        
        Args:
            key: Key name (e.g., 'enter', 'space', 'a')
            duration: How long to hold the key
        """
        if not self.pyautogui:
            print(f"Would press key: {key}")
            return
        
        self.pyautogui.press(key)
        time.sleep(duration)
    
    def type_text(self, text: str, interval: float = 0.05):
        """
        Type text.
        
        Args:
            text: Text to type
            interval: Interval between keystrokes
        """
        if not self.pyautogui:
            print(f"Would type: {text}")
            return
        
        self.pyautogui.write(text, interval=interval)


class ActionMapper:
    """Maps poker actions to screen coordinates."""
    
    def __init__(self):
        """Initialize action mapper with default coordinates."""
        # Default coordinates for common poker sites (as percentages)
        self.default_positions = {
            'fold': (0.421, 0.86),
            'check': (0.556, 0.86),
            'call': (0.556, 0.86),
            'raise': (0.691, 0.86),
            'bet_slider': (0.556, 0.78)
        }
        
        self.controller = ScreenController()
    
    def execute_action(self, action: str, raise_amount: Optional[int] = None):
        """
        Execute a poker action on screen.
        
        Args:
            action: Action name ('fold', 'check', 'call', 'raise')
            raise_amount: Amount to raise (if applicable)
        """
        action = action.lower()
        
        if action not in self.default_positions:
            print(f"Unknown action: {action}")
            return
        
        x, y = self.default_positions[action]
        
        print(f"Executing action: {action}")
        self.controller.click_at_percentage(x, y)
        
        # Handle raise amount if needed
        if action == 'raise' and raise_amount is not None:
            time.sleep(0.5)
            # Click bet slider or input field
            self.controller.type_text(str(raise_amount))
            time.sleep(0.3)
            self.controller.press_key('enter')
    
    def update_position(self, action: str, x_percent: float, y_percent: float):
        """
        Update screen position for an action.
        
        Args:
            action: Action name
            x_percent: X position as percentage
            y_percent: Y position as percentage
        """
        self.default_positions[action] = (x_percent, y_percent)
