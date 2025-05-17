import functools
import cProfile
import pstats
import io
from contextlib import contextmanager

# Profiling globals
_profiler = None

def start_profiling():
    """Start profiling session"""
    global _profiler
    _profiler = cProfile.Profile()
    _profiler.enable()
    return _profiler

def stop_profiling(print_stats=True, sort_by='time', lines=20):
    """Stop profiling and optionally print stats"""
    global _profiler
    if _profiler is not None:
        _profiler.disable()
        if print_stats:
            s = io.StringIO()
            ps = pstats.Stats(_profiler, stream=s).sort_stats(sort_by)
            ps.print_stats(lines)
            print(s.getvalue())
        return _profiler
    return None

@contextmanager
def profile_section(section_name):
    """Context manager for profiling specific code sections"""
    global _profiler
    # Check if global profiler is already running
    if _profiler is not None:
        # If global profiler is active, just yield without profiling
        yield
        return

    # Otherwise, profile the section
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    print(f"\n--- Profiling results for {section_name} ---")
    ps.print_stats(20)
    print(s.getvalue())

def profile(func):
    """Decorator to profile a function"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _profiler
        # Check if global profiler is already running
        if _profiler is not None:
            # If global profiler is active, just run the function without profiling
            return func(*args, **kwargs)

        # Otherwise, profile the function
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        print(f"\n--- Profiling results for {func.__name__} ---")
        ps.print_stats(20)
        print(s.getvalue())
        return result
    return wrapper
