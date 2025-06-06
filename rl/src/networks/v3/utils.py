from .encoder import StateEncoder, StateEncoderConfig, MapEncoderConfig, MapStateEncoder, RecurrentMapStateEncoderConfig, RecurrentMapStateEncoder
from ..v2.ppo import DiscretePolicy, DiscretePolicyConfig, ValueNetwork, ValueNetworkConfig, PPOActorCritic


def initialize_model(
    encoder_config: StateEncoderConfig | MapEncoderConfig | RecurrentMapStateEncoderConfig,
    actor_config: DiscretePolicyConfig,
    critic_config: ValueNetworkConfig,
    device=None,
    **kwargs
):
    if isinstance(encoder_config, StateEncoderConfig):
        actor_encoder = StateEncoder(encoder_config)
        critic_encoder = StateEncoder(encoder_config)
    elif isinstance(encoder_config, RecurrentMapStateEncoderConfig):
        actor_encoder = RecurrentMapStateEncoder(encoder_config)
        critic_encoder = RecurrentMapStateEncoder(encoder_config)
    else:
        actor_encoder = MapStateEncoder(encoder_config)
        critic_encoder = MapStateEncoder(encoder_config)

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
