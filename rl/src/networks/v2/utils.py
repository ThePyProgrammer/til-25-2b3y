from .encoder import MapEncoder, MapEncoderConfig
from .ppo import DiscretePolicy, DiscretePolicyConfig, ValueNetwork, ValueNetworkConfig, PPOActorCritic


def initialize_model(
    encoder_config: MapEncoderConfig,
    actor_config: DiscretePolicyConfig,
    critic_config: ValueNetworkConfig,
    device=None,
    **kwargs
):
    actor_encoder = MapEncoder(encoder_config)
    critic_encoder = MapEncoder(encoder_config)
    actor = DiscretePolicy(actor_config)
    critic = ValueNetwork(critic_config)

    model = PPOActorCritic(
        actor_encoder,
        critic_encoder,
        actor,
        critic
    )

    if device is not None:
        model.to(device)

    return model
