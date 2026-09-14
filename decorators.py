import time
import functools
from typing import Callable

# Decorator that logs how long the wrapped function took to run
def log_execution_time(view_func: Callable) -> Callable:
    @functools.wraps(view_func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = view_func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[{view_func.__name__}] finished in {elapsed:.2f}s")
        return result
    return wrapper