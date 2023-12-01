import torch


class Architecture(torch.nn.Module):
    """
    All architectures should accept an argument
    `num_ifos` as their first argument, since the
    CLI links to this argument from the datamodule
    at initialization time.

    TODO: if you really wanted to enforce this with
    gross overkill, you could make a metaclass
    that inspects the __init__ method of each new
    subclass and ensures that the first parameter
    is an int called `num_ifos`.
    """

    # def __init__(self, num_ifos: int) -> None:
    #     pass
