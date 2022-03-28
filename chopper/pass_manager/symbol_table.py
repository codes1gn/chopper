import mlir.astnodes as astnodes
from typing import Optional

__all__ = [
    "FeedForwardSymbolTable",
    "feed_forward_symbol_table",
    "AutodiffSymbolTable",
    "autodiff_symbol_table",
]


class AutodiffSymbolTable(object):
    __slots__ = [
        "type_table",
        "value_table",
        "autodiff_tree",
    ]

    def __init__(self):
        self.type_table = {}
        self.value_table = {}
        _name = None
        _attributes = None
        _out_block = astnodes.Block(label=None, body=[None])
        _out_region = astnodes.Region(body=[_out_block])
        _module = astnodes.Module(name=_name, attributes=_attributes, region=_out_region, location=None)
        _mlirfile = astnodes.MLIRFile(definitions=[], modules=[_module])
        self.autodiff_tree = _mlirfile

    def get_autodiff_graph(self) -> astnodes.MLIRFile:
        return self.autodiff_tree

    def reset_autodiff_graph(self):
        self.autodiff_tree.modules[0].region.body[0].body = [None]

    def set_autodiff_graph(self, func: astnodes.Function):
        self.autodiff_tree.modules[0].region.body[0].body = [func]

    def insert(self, query_name: str, query_type: astnodes.Type):
        self.type_table[query_name] = query_type
        self.value_table[query_name] = astnodes.SsaId(value=query_name, op_no=None)

    def reset_symbol_table(self):
        self.type_table = {}
        self.value_table = {}

    # lookup method to query 1 of 3 value forms in:
    # value: SsaId | type: Type | typed-value: NamedArgument
    def lookup(self, query_name: str, query_key: str) -> Optional[astnodes.Type]:
        if query_key == "value":
            return self.value_table.get(query_name)
        elif query_key == "type":
            return self.type_table.get(query_name)
        elif query_key == "typed-value":
            return astnodes.NamedArgument(name=self.value_table.get(query_name), type=self.type_table.get(query_name))
        else:
            assert 0, "lookup for non-exist query key"

    def __str__(self) -> str:
        debug_str = ""
        debug_str += ">=============================<\n"
        debug_str += ">==== SymbolTable Summary ====<\n"
        debug_str += ">=============================<\n\n"
        debug_str += "Count of Symbol Entries = {}\n".format(len(self.value_table))
        debug_str += "Listing Symbol Entries ...\n\n"
        iid = 0
        for key, value in self.value_table.items():
            debug_str += "\nSymbol #{} \n=> {} \n=> {}\n".format(iid, value, self.lookup(key, "type"))
            iid += 1
        debug_str += "\n\n>=============================<\n"
        debug_str += ">== End SymbolTable Summary ==<\n"
        debug_str += ">=============================<\n"
        return debug_str


class FeedForwardSymbolTable(object):
    __slots__ = [
        "value_table",
        "type_table",
        "pass_again",
    ]

    def __init__(self):
        self.type_table = {}
        self.value_table = {}
        self.pass_again = True

    # store ssaid and type
    def insert(self, query_name: str, query_type: astnodes.Type):
        self.type_table[query_name] = query_type
        self.value_table[query_name] = astnodes.SsaId(value=query_name, op_no=None)

    def reset_symbol_table(self):
        self.type_table = {}
        self.value_table = {}

    # lookup method to query 1 of 3 value forms in:
    # value: SsaId | type: Type | typed-value: NamedArgument
    def lookup(self, query_name: str, query_key: str) -> Optional[astnodes.Type]:
        if query_key == "value":
            return self.value_table.get(query_name)
        elif query_key == "type":
            return self.type_table.get(query_name)
        elif query_key == "typed-value":
            return astnodes.NamedArgument(name=self.value_table.get(query_name), type=self.type_table.get(query_name))
        else:
            assert 0, "lookup for non-exist query key"

    def __str__(self) -> str:
        debug_str = ""
        debug_str += ">=============================<\n"
        debug_str += ">==== SymbolTable Summary ====<\n"
        debug_str += ">=============================<\n\n"
        debug_str += "Count of Symbol Entries = {}\n".format(len(self.value_table))
        debug_str += "Listing Symbol Entries ...\n\n"
        iid = 0
        for key, value in self.value_table.items():
            debug_str += "\nSymbol #{} \n=> {} \n=> {}\n".format(iid, value, self.lookup(key, "type"))
            iid += 1
        debug_str += "\n\n>=============================<\n"
        debug_str += ">== End SymbolTable Summary ==<\n"
        debug_str += ">=============================<\n"
        return debug_str


# TODO make it singleton
feed_forward_symbol_table = FeedForwardSymbolTable()
autodiff_symbol_table = AutodiffSymbolTable()
