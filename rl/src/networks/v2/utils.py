from .encoder import MapEncoder, MapEncoderConfig, TemporalMapEncoder, TemporalMapEncoderConfig
from .ppo import DiscretePolicy, DiscretePolicyConfig, ValueNetwork, ValueNetworkConfig, PPOActorCritic


def initialize_model(
    encoder_config: MapEncoderConfig | TemporalMapEncoderConfig,
    actor_config: DiscretePolicyConfig,
    critic_config: ValueNetworkConfig,
    device=None,
    **kwargs
):
    if isinstance(encoder_config, TemporalMapEncoderConfig):
        actor_encoder = TemporalMapEncoder(encoder_config)
        critic_encoder = TemporalMapEncoder(encoder_config)
    else:
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
