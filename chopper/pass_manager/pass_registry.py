from chopper.scaffold.utils import memoized_classproperty
from .passes import *


# lazy load the pass register
class PassRegistry(object):
    @memoized_classproperty
    def pass_table(cls) -> dict:
        """register passes into registry, will only run once

        Returns:
            dict: pass table that pass name is key and pass str(id) is value
        """

        # TODO(albert) add more passes
        _pass_table = {}

        def register_pass(pass_class):
            nonlocal _pass_table
            _pass_table[pass_class.__name__] = str(register_pass.id_cnt)
            register_pass.id_cnt += 1

        # register passes
        register_pass.id_cnt = 0
        register_pass(IdenticalPastPass)
        register_pass(StatementConversionPass)
        register_pass(AnnotateCompletionPass)
        print("_pass_table == > ", _pass_table)
        return _pass_table
