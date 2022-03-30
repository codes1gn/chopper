from typing import Callable, List, Optional, Tuple, NamedTuple
import torch


__all__ = [
    "TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME",
    "unique_module_name",
    "ArgAnnotation",
]

# TODO move this hardcode into base
TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME = "_torch_dsl_arg_annotations"
ArgAnnotation = Tuple[List[int], torch.dtype]

# compile name prefix of module names
class UniqueModuleName:
    def __init__(self):
        self.forward_name = None
        self.backward_name = None

    def set_forward(self, name):
        self.forward_name = name

    def set_backward(self, name):
        self.backward_name = name

    def get_forward(self):
        return self.forward_name

    def get_backward(self):
        return self.backward_name


unique_module_name = UniqueModuleName()
