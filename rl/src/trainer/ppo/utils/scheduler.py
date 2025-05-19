from typing import Any

import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim

def create_scheduler(optimizer: optim.Optimizer, args: Any, total_steps: int):
    """
    Creates a learning rate scheduler based on parsed arguments.

    Args:
        optimizer: The optimizer for which to create the scheduler.
        args: Parsed arguments containing scheduler configuration (--sched, --lr, --timesteps).
        total_steps: Total number of training steps (timesteps).

    Returns:
        A PyTorch learning rate scheduler instance or None if no scheduler is specified.
    """
    scheduler_type = args.sched.lower()

    if scheduler_type == 'cosine':
        # Cosine Annealing without restart
        # T_max is half a period, usually set to total steps for one full cycle
        # Or set to total_steps // args.epochs if scheduling per epoch
        # Since we are training per timestep/episode and want a single schedule over total timesteps:
        t_max = total_steps # Schedule over the total number of timesteps
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        print(f"Using Cosine Annealing LR scheduler with T_max={t_max}")
        return scheduler

    elif scheduler_type == 'linear':
        # Linear scheduler (e.g., linear warmup and decay)
        # This often requires defining specific milestones or steps.
        # A simple linear decay from initial LR to 0 over total steps:
        def linear_lr(step):
            if step < 0: # Handle edge case
                return 1.0
            return max(0.0, 1.0 - (step / total_steps))

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_lr)
        print(f"Using Linear LR scheduler over {total_steps} steps.")
        return scheduler

    elif scheduler_type == 'none':
        print("No learning rate scheduler used.")
        return None

    else:
        print(f"Warning: Unknown scheduler type '{args.sched}'. No scheduler used.")
        return None

if __name__ == '__main__':
    # Example usage (requires a dummy optimizer and args)
    print("Testing create_scheduler function (dummy data)...")

    # Dummy optimizer
    dummy_model_param = [torch.randn(10, 10, requires_grad=True)]
    dummy_optimizer = optim.Adam(dummy_model_param, lr=0.001)

    # Dummy args
    class DummyArgs:
        def __init__(self, sched='none', lr=0.001, timesteps=10000):
            self.sched = sched
            self.lr = lr
            self.timesteps = timesteps

    total_steps = 10000 # Corresponds to dummy_args.timesteps

    # Test 'none'
    args_none = DummyArgs(sched='none', timesteps=total_steps)
    scheduler_none = create_scheduler(dummy_optimizer, args_none, total_steps)
    print(f"Scheduler for 'none': {scheduler_none}")

    # Test 'cosine'
    args_cosine = DummyArgs(sched='cosine', timesteps=total_steps)
    scheduler_cosine = create_scheduler(dummy_optimizer, args_cosine, total_steps)
    print(f"Scheduler for 'cosine': {scheduler_cosine}")

    # Test 'linear'
    args_linear = DummyArgs(sched='linear', timesteps=total_steps)
    scheduler_linear = create_scheduler(dummy_optimizer, args_linear, total_steps)
    print(f"Scheduler for 'linear': {scheduler_linear}")

    # Test unknown
    args_unknown = DummyArgs(sched='unknown', timesteps=total_steps)
    scheduler_unknown = create_scheduler(dummy_optimizer, args_unknown, total_steps)
    print(f"Scheduler for 'unknown': {scheduler_unknown}")

    # Example of stepping a scheduler
    if scheduler_cosine:
         print("\nStepping Cosine scheduler:")
         initial_lr = dummy_optimizer.param_groups[0]['lr']
         print(f"Initial LR: {initial_lr}")
         for step in range(10):
             scheduler_cosine.step(step) # CosineAnnealingLR steps based on epoch/step count
             current_lr = dummy_optimizer.param_groups[0]['lr']
             print(f"Step {step}, LR: {current_lr}")

    # Reset optimizer LR for linear test
    dummy_optimizer.param_groups[0]['lr'] = args_linear.lr
    if scheduler_linear:
         print("\nStepping Linear scheduler:")
         initial_lr = dummy_optimizer.param_groups[0]['lr']
         print(f"Initial LR: {initial_lr}")
         for step in range(10):
             scheduler_linear.step(step) # LambdaLR steps based on epoch/step count
             current_lr = dummy_optimizer.param_groups[0]['lr']
             print(f"Step {step}, LR: {current_lr}")
