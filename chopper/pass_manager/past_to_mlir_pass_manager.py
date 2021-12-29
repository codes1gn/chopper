from chopper.pass_manager.pass_manager_base import PassManagerBase
from chopper.pass_manager.passes import IdenticalPastPass, StatementConversionPass

__all__ = [
    "PastToMlirPassManager",
]


class PastToMlirPassManager(PassManagerBase):
    """The class inherit PassManagerBase that to register passes.

    Attributions:
        None.
    """
    def register_passes(self):
        """Register passes via add_pass func in PassManagerBase.
        """
        self.add_pass(IdenticalPastPass)
        self.add_pass(StatementConversionPass)

        return
