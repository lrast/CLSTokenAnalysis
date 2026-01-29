# Shared utilities

class HookContext:
    """ Context that clears hook when it is done """
    def __init__(self, hook_handles):
        self.hook_handles = hook_handles

    def cleanup(self):
        """Explicitly remove the hook. Should be called when done."""
        for name, handle in self.hook_handles.items():
            handle.remove()

    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup the hook"""
        self.cleanup()
        return False

    def __del__(self):
        """Cleanup the hook as fallback"""
        self.cleanup()
