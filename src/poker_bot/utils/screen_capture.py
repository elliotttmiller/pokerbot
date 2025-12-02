"""
Screen Capture Utility.

Functions for taking high-resolution screenshots of the poker table.
"""

import os
from typing import Optional, Tuple

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def capture_screen(region: Optional[Tuple[int, int, int, int]] = None,
                   output_path: Optional[str] = None) -> Optional['Image.Image']:
    """
    Capture screenshot of screen or specified region.
    
    Args:
        region: Optional (x, y, width, height) tuple for region capture
        output_path: Optional path to save screenshot
        
    Returns:
        PIL Image object, or None if capture failed
    """
    if not PYAUTOGUI_AVAILABLE:
        print("[ScreenCapture] PyAutoGUI not available")
        return None
    
    try:
        if region:
            screenshot = pyautogui.screenshot(region=region)
        else:
            screenshot = pyautogui.screenshot()
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            screenshot.save(output_path)
            print(f"[ScreenCapture] Saved to {output_path}")
        
        return screenshot
        
    except Exception as e:
        print(f"[ScreenCapture] Error: {e}")
        return None


def capture_window(window_title: str,
                   output_path: Optional[str] = None) -> Optional['Image.Image']:
    """
    Capture screenshot of a specific window.
    
    Args:
        window_title: Title (or partial title) of window to capture
        output_path: Optional path to save screenshot
        
    Returns:
        PIL Image object, or None if window not found
    """
    try:
        import pygetwindow as gw
        
        # Find window
        windows = gw.getWindowsWithTitle(window_title)
        if not windows:
            print(f"[ScreenCapture] Window not found: {window_title}")
            return None
        
        window = windows[0]
        
        # Get window bounds
        region = (window.left, window.top, window.width, window.height)
        
        return capture_screen(region, output_path)
        
    except ImportError:
        print("[ScreenCapture] pygetwindow not available, using full screen")
        return capture_screen(output_path=output_path)
    except Exception as e:
        print(f"[ScreenCapture] Error: {e}")
        return None


def get_poker_window_region(window_title: str = "PokerStars") -> Optional[Tuple[int, int, int, int]]:
    """
    Get the screen region of a poker client window.
    
    Args:
        window_title: Title to search for
        
    Returns:
        Tuple of (x, y, width, height) or None
    """
    try:
        import pygetwindow as gw
        
        windows = gw.getWindowsWithTitle(window_title)
        if not windows:
            return None
        
        window = windows[0]
        return (window.left, window.top, window.width, window.height)
        
    except Exception:
        return None


__all__ = ['capture_screen', 'capture_window', 'get_poker_window_region']
