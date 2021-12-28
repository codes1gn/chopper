def _load_extension():
    # TODO: Remote the RTLD_GLOBAL hack once local, cross module imports
    # resolve symbols properly. Something is keeping the dynamic loader on
    # Linux from treating the following vague symbols as the same across
    # _mlir and _npcomp:
    #   mlir::detail::TypeIDExported::get<mlir::FuncOp>()::instance
    import sys
    import ctypes

    flags = sys.getdlopenflags()
    sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)
    import chopper_compiler_module

    sys.setdlopenflags(flags)

    # import mlir
    # mlir._cext.globals.append_dialect_search_prefix("npcomp.dialects")
    return chopper_compiler_module


# TODO need to remove this part at parent level and only keep one
chopper_compiler_ext = _load_extension()
# chopper_compiler_ext._register_all_passes()
# Top-level symbols.
# from .exporter import *
# from .types import *

# from . import tracing
# from . import utils
from .python_jit_runner import *
