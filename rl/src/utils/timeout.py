import threading

class TimeoutError(Exception):
    pass

def timeout(timeout_seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Store the result
            result = None
            exception = None

            # Define the thread function
            def worker():
                nonlocal result, exception
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    exception = e

            # Create and start the thread
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()

            # Wait for completion or timeout
            thread.join(timeout_seconds)
            if thread.is_alive():
                # Thread is still running after timeout
                raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")

            # If there was an exception in the thread, raise it
            if exception:
                raise exception

            return result
        return wrapper
    return decorator
