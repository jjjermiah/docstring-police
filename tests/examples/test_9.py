# test_9.py
def external_func():
    """
    This function references another function in its docstring.
    
    See also
    --------
    internal_func : Helper function used within this one.
    """
    return internal_func()

def internal_func():
    """Simple internal function."""
    return "Internal"
