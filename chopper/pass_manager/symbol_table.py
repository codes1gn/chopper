import mlir.astnodes as astnodes
from typing import Optional

__all__ = [
    "SymbolTable",
    "global_symbol_table",
]


# TODO debug print
class SymbolEntry(object):

    __slots__ = [
        "name",
        "mlirtype",
    ]

    def __init__(self, name: str, mlirtype: astnodes.Type):
        self.name = name
        self.mlirtype = mlirtype

    def debug_str(self) -> str:
        return "\n    obj={},\n    name={},\n    type={}\n".format(self, self.name, self.mlirtype)

    def get_name(self) -> str:
        return self.name

    def get_type(self) -> astnodes.Type:
        return self.mlirtype


class SymbolTable(object):
    __slots__ = [
        "scoped_symbol_table",
        "pass_again",
    ]

    def __init__(self):
        self.scoped_symbol_table = {}
        self.pass_again = True

    def register_symbol(self, symbol_entry: SymbolEntry):
        self.scoped_symbol_table[symbol_entry.name] = symbol_entry

    def reset_symbol_table(self):
        self.scoped_symbol_table = {}

    def query(self, key: str) -> Optional[SymbolEntry]:
        return self.scoped_symbol_table.get(key)

    def __str__(self) -> str:
        debug_str = ""
        debug_str += ">=============================<\n"
        debug_str += ">==== SymbolTable Summary ====<\n"
        debug_str += ">=============================<\n\n"
        debug_str += "Count of Symbol Entries = {}\n".format(len(self.scoped_symbol_table))
        debug_str += "Listing Symbol Entries ...\n\n"
        iid = 0
        for key, value in self.scoped_symbol_table.items():
            debug_str += "Symbol Entry #{} =>{}".format(iid, value.debug_str())
            iid += 1
        debug_str += "\n\n>=============================<\n"
        debug_str += ">== End SymbolTable Summary ==<\n"
        debug_str += ">=============================<\n"
        return debug_str


# TODO make it singleton
global_symbol_table = SymbolTable()
