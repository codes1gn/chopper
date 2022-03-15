from typing import Callable, List, Optional, Tuple, NamedTuple
import torch


__all__ = [
    'TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME',
    'ArgAnnotation',
]

# TODO move this hardcode into base
TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME = '_torch_dsl_arg_annotations'
ArgAnnotation = Tuple[List[int], torch.dtype]

