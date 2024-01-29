from flax import linen as nn  # Linen API


class NeuralOdeBase(nn.Module):
    def forward_dynamics(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)
