from collections import OrderedDict
from functools import partial

import diffusers
import torch
import torch.nn as nn

from train_methods.mce_models.hooks.base import BaseHooker


class FeedForwardHooker(BaseHooker):
    def __init__(
        self,
        pipeline: nn.Module,
        regex: str | list[str],
        masking: str,
        dst: str,
        epsilon: float = 0.0,
        eps: float = 1e-6,
        use_log: bool = False,
        binary: bool = False,
    ):
        self.pipeline = pipeline
        self.net = pipeline.unet if hasattr(pipeline, "unet") else pipeline.transformer
        self.regex = [regex] if isinstance(regex, str) else regex
        self.hook_dict = {}
        self.masking = masking
        self.dst = dst
        self.epsilon = epsilon
        self.eps = eps
        self.use_log = use_log
        self.lambs: list[torch.Tensor | None] = []
        self.lambs_module_names = []  # store the module names for each lambda block
        self.hook_counter = 0
        self.module_neurons = OrderedDict()
        self.binary = binary  # default, need to discuss if we need to keep this attribute or not

    def add_hooks_to_ff(self, hook_fn: callable):
        """
        Add forward hooks to every feed forward layer matching the regex
        :param hook_fn: a callable to be added to torch nn module as a hook
        :return: dictionary of added hooks
        """
        total_hooks = 0
        for name, module in self.net.named_modules():
            name_last_word = name.split(".")[-1]
            if "ff" in name_last_word:
                if self.regular_expression_match(self.regex, name):
                    hook_fn_with_name = partial(hook_fn, name=name)
                    # only apply hook on act_fn/module
                    # SDXL using GEGLU
                    # SD3, FLUX use gelu-approximation

                    actual_module = module.net[0]
                    hook = actual_module.register_forward_hook(hook_fn_with_name, with_kwargs=True)
                    self.hook_dict[name] = hook

                    if isinstance(actual_module, diffusers.models.activations.GEGLU):  # geglu
                        # due to the GEGLU chunking, we need to divide by 2
                        self.module_neurons[name] = actual_module.proj.out_features // 2
                    elif isinstance(actual_module, diffusers.models.activations.GELU):  # gelu
                        self.module_neurons[name] = actual_module.proj.out_features
                    else:
                        raise NotImplementedError(f"Module {name} is not implemented, please check")
                    print(f"Adding hook to {name}, neurons: {self.module_neurons[name]}")
                    total_hooks += 1
        print(f"Total hooks added: {total_hooks}")
        return self.hook_dict

    def add_hooks(self, init_value=1.0):
        hook_fn = self.get_ff_masking_hook(init_value)
        self.add_hooks_to_ff(hook_fn)
        # initialize the lambda
        self.lambs = [None] * len(self.hook_dict)
        # initialize the lambda module names
        self.lambs_module_names = [None] * len(self.hook_dict)

    def masking_fn(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states_dtype = hidden_states.dtype
        mask = self.base_masking_fn(**kwargs)
        epsilon = kwargs.get("epsilon", 0.0)
        assert isinstance(epsilon, float)
        hidden_states = hidden_states * mask + torch.randn_like(hidden_states) * epsilon * (1 - mask)
        return hidden_states.to(hidden_states_dtype)

    def get_ff_masking_hook(self, init_value=1.0):
        """
        Get a hook function to mask feed forward layer
        """

        def hook_fn(module, args, kwargs, output, name):
            # initialize lambda with acual head dim in the first run
            if self.lambs[self.hook_counter] is None:
                self.lambs[self.hook_counter] = (
                    torch.ones(self.module_neurons[name], device=self.pipeline.device) * init_value
                )
                self.lambs[self.hook_counter].requires_grad = True
                # load ff lambda module name for logging
                self.lambs_module_names[self.hook_counter] = name

            # perform masking
            output = self.masking_fn(
                output,
                masking=self.masking,
                lamb=self.lambs[self.hook_counter],
                epsilon=self.epsilon,
                eps=self.eps,
                use_log=self.use_log,
            )
            self.hook_counter += 1
            self.hook_counter %= len(self.lambs)
            return output

        return hook_fn
