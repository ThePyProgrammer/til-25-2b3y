import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Train a PPO agent for the scout.')

    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--gae', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip_eps', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='PPO entropy coefficient')
    parser.add_argument('--value_loss_coef', type=float, default=0.5, help='PPO value loss coefficient')

    parser.add_argument('--epochs', type=int, default=10, help='number of PPO epochs per update')
    parser.add_argument('--timesteps', type=int, default=1_000_000, help='total timesteps to train')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size for PPO update')
    parser.add_argument('--episodes_per_update', type=int, default=1, help='number of episodes to collect before PPO update')

    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam', 'adamw'], help='optimizer')
    parser.add_argument('--sched', type=str, default='none', help='learning rate scheduler (none, cosine, linear, etc.) - not implemented yet')
    parser.add_argument('--bfloat16', action='store_true', help='use bfloat16 for training')

    parser.add_argument('--resume', action='store_true', help='resume training from save_dir')
    parser.add_argument('--save_dir', type=str, default='./ppo_scout_train', help='directory to save model and logs')
    parser.add_argument('--save_interval', type=int, default=5000, help='saving interval')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--env_id', type=str, default='gridworld', help='environment id') # Keep flexibility for different envs
    parser.add_argument('--render', action='store_true', help='render environment during training')
    parser.add_argument('--num_guards', type=int, default=3, help='number of guards')

    args = parser.parse_args()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    return args

if __name__ == '__main__':
    # Example usage
    args = parse_args()
    print(args)
