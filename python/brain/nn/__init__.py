"""
Neural network layers for Brain.
"""
try:
    from ..brain_native import Linear, Conv2d, LayerNorm, MSELoss, CrossEntropyLoss, Value, Tensor
except ImportError:
    pass

class Module:
    """
    Base class for all neural network modules.
    Your models should subclass this class.
    """
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("forward method must be implemented by subclass")

    def parameters(self):
        """
        Returns an iterator/list over module parameters.
        """
        params = []
        for name, val in self.__dict__.items():
            if isinstance(val, Module) or hasattr(val, "parameters"):
                params.extend(val.parameters())
            elif isinstance(val, (Value, Tensor)) and getattr(val, "requires_grad", False):
                params.append(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, Module) or hasattr(item, "parameters"):
                        params.extend(item.parameters())
                    elif isinstance(item, (Value, Tensor)) and getattr(item, "requires_grad", False):
                        params.append(item)
        return params

    def zero_grad(self):
        """
        Sets gradients of all model parameters to zero.
        """
        for p in self.parameters():
            p.zero_grad()

__all__ = ["Module", "Linear", "Conv2d", "LayerNorm", "MSELoss", "CrossEntropyLoss"]
