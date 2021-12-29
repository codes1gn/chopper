from typing import Any
from chopper.scaffold.utils import *


class PassBase(object):
    """The Based Pass class.

    Attributions:
        None.
    """
    def __init__(self):
        """initilize the PassBase class.
        """
        pass

    def run_pass(self, _cnode: Any) -> Any:
        """The based function to run the pass to genenrate textual IR

        Args:
            _cnode (Any): the source code astnode.

        Returns:
            Any : the source code astnode after run each pass.
        """
        return _cnode
