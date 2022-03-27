import ast
import torch
import astunparse

from chopper.scaffold.utils import *
from mlir import astnodes
from .node_visitor_base import NodeVisitorBase
from chopper.pass_manager.symbol_table import global_symbol_table, SymbolTable, SymbolEntry
from mlir.astnodes import (
    CustomOperation,
    FunctionType,
    NamedArgument,
    Dimension,
    RankedTensorType,
    NoneType,
    FloatTypeEnum,
    FloatType,
)

MlirSsaId = astnodes.SsaId

__all__ = [
    "AnnotateTypesVisitor",
]


class AnnotateTypesVisitor(NodeVisitorBase):
    """This is class that identical transformer python native Call astnode.

    Attributes:
        None.
    """

    __slots__ = []

    def __init__(self, arg_annotation: ArgAnnotation):
        """Initialize StmtConversionReadyCheckVisitor class via inherit NodeVisitorBase."""

        super().__init__()
        self.arg_annotation = arg_annotation

    def visit_FunctionDef(self, node: ast.AST) -> ast.AST:
        """Annotate function arguments types according to the annotation
        config passed in from the __init__ method.

        Args:
            node (ast.AST): FunctionDef with corresponding mlir astnode attribution.

        Returns:
            ast.AST: FunctionDef with corresponding mlir astnode attribution.
        """
        print(self.__str__(), "::visit_FunctionDef\n")
        # only handles arguments > 0
        assert len(node.args.args) > 0
        # WORKAROUND
        setattr(node, TORCH_MLIR_ARG_ANNOTATIONS_ATTR_NAME, self.arg_annotation)
        func_args = node.args.args
        # HARDCODE TYPE DEFINE FOR AUTODIFF FUNCTION RETURN
        # at bp path, function outputs is equivalent to input arguments
        func_return_for_autodiff = []
        activation_save_for_autodiff = []
        for arg_index in range(len(func_args)):
            _argname = func_args[arg_index].arg
            # TODO move this hardcode into base
            # case 1, arguments is float type
            if global_symbol_table.lookup(_argname) is not None:
                continue

            if self.arg_annotation[arg_index] is None:
                _type = NoneType()
            else:
                # TODO use From Trait or other way to make this conversion elegant
                if self.arg_annotation[arg_index][1] is torch.float32:
                    _dtype = FloatType(FloatTypeEnum.f32)
                else:
                    assert 0, "Not support this dtype for annotation"
                _dim = [
                    Dimension(self.arg_annotation[arg_index][0][k])
                    for k in range(len(self.arg_annotation[arg_index][0]))
                ]
                _type = RankedTensorType(
                    dimensions=_dim,
                    element_type=_dtype,
                )
                # record this type for autodiff use only if it is not None
                func_return_for_autodiff.append(_type)

                # record save for backward activations as arguments of bp compute
                _activation_arg = NamedArgument(name=MlirSsaId(value=_argname + "_activation", op_no=None), type=_type)
                activation_save_for_autodiff.append(_activation_arg)
            # record this type for symbol table use even it is NoneType of self
            global_symbol_table.insert(SymbolEntry(_argname, _type))

        global_symbol_table.insert(SymbolEntry("AutodiffFuncReturnType", func_return_for_autodiff))
        global_symbol_table.insert(SymbolEntry("ActivationSaveForAutodiff", activation_save_for_autodiff))

        super().generic_visit(node)
        return node

    def visit_Assign(self, node: ast.AST) -> ast.AST:
        """handle typing annotation and check, infer the operand and return
        types by lookuping symbol table; and create corresponding symbol
        entries into symbol table.

        Args:
            node (ast.AST): AssignOp in python ast.AST.

        Returns:
            ast.AST: same type as input.
        """
        print(self.__str__(), "::visit_Assign\n")
        _ret_op = node.targets
        _rhs_stmt = node.value
        if isinstance(_rhs_stmt, ast.BinOp):
            # handle c = a + b where '+' is a BinOp
            _lhs_arg = _rhs_stmt.left
            _rhs_arg = _rhs_stmt.right
            _opcode = _rhs_stmt.op

            # VERIFY SYMBOL TABLE IF READY FOR THIS OP
            _lhs_sym_entry = global_symbol_table.lookup(_lhs_arg.id)
            _rhs_sym_entry = global_symbol_table.lookup(_rhs_arg.id)
            if _lhs_sym_entry is None or _rhs_sym_entry is None:
                global_symbol_table.pass_again = True
                super().generic_visit(node)
                return node

            # HANDLE OPERAND SHAPE
            _lhs_type = _lhs_sym_entry.get_type()
            _rhs_type = _rhs_sym_entry.get_type()
            assert _lhs_type.element_type == _rhs_type.element_type
            _lhs_shape = _lhs_type.dimensions
            _rhs_shape = _rhs_type.dimensions
            assert _lhs_shape == _rhs_shape, "expected same shape of lhs and rhs arguments"
            for _ret_op_element in _ret_op:
                _ret_sym_entry = SymbolEntry(_ret_op_element.id, _lhs_type)
                global_symbol_table.insert(_ret_sym_entry)

        elif isinstance(_rhs_stmt, ast.Call):
            print(astunparse.dump(_rhs_stmt))
            # TODO change this way of matching into String Utils
            # e.g. torch.nn.functional.linear and
            # torch.matmul can be explored equally
            # _call_lib = _rhs_stmt.func.value.id  # torch
            # assert _rhs_stmt.func.value.id == "torch", "Found function call other than Torch DSL"

            _call_method = _rhs_stmt.func.attr  # exp or add
            _args = _rhs_stmt.args  # ast.Name, ast.Name
            _arg_type_entries = [global_symbol_table.lookup(_argname.id) for _argname in _args]

            # if any arg types are not inferred, means the infer of this call op is not ready
            # run the pass again
            for _ in _arg_type_entries:
                if _ is None:
                    global_symbol_table.pass_again = True
                    super().generic_visit(node)
                    return node

            # handle accepted function calls and assert for guardian
            # TODO build a builder or conversion or mapping utils
            if _call_method == "exp" or _call_method == "tanh":

                assert len(_arg_type_entries) == 1, "expected unary, too long of arguments for unaryop call"
                _result_type = _arg_type_entries[0].get_type()
                for _ret_op_element in _ret_op:
                    _ret_sym_entry = SymbolEntry(_ret_op_element.id, _result_type)
                    global_symbol_table.insert(_ret_sym_entry)

            elif _call_method == "add" or _call_method == "sub" or _call_method == "mul":

                print(_call_method)
                assert len(_arg_type_entries) == 2, "expected binary, too long of arguments for unaryop call"
                _lhs_type = _arg_type_entries[0].get_type()
                _rhs_type = _arg_type_entries[1].get_type()
                assert _lhs_type.element_type == _rhs_type.element_type
                _lhs_shape = _lhs_type.dimensions
                _rhs_shape = _rhs_type.dimensions
                assert _lhs_shape == _rhs_shape, "expected same shape of lhs and rhs arguments"
                for _ret_op_element in _ret_op:
                    _ret_sym_entry = SymbolEntry(_ret_op_element.id, _lhs_type)
                    global_symbol_table.insert(_ret_sym_entry)
            elif _call_method == "linear" or _call_method == "matmul":
                assert len(_arg_type_entries) == 2, "expected binary, too long of arguments for unaryop call"
                _lhs_type = _arg_type_entries[0].get_type()
                _rhs_type = _arg_type_entries[1].get_type()
                assert _lhs_type.element_type == _rhs_type.element_type
                _lhs_shape = _lhs_type.dimensions
                _rhs_shape = _rhs_type.dimensions
                # anchor
                _ret_type = RankedTensorType(
                    dimensions=[
                        _lhs_shape[0],
                        _rhs_shape[1],
                    ],
                    element_type=_lhs_type.element_type,
                )

                for _ret_op_element in _ret_op:
                    _ret_sym_entry = SymbolEntry(_ret_op_element.id, _ret_type)
                    global_symbol_table.insert(_ret_sym_entry)
            elif _call_method == "conv2d":
                assert len(_arg_type_entries) == 2, "expected binary, too long of arguments for unaryop call"
                _lhs_type = _arg_type_entries[0].get_type()
                _rhs_type = _arg_type_entries[1].get_type()
                assert _lhs_type.element_type == _rhs_type.element_type
                _lhs_shape = _lhs_type.dimensions
                _rhs_shape = _rhs_type.dimensions
                # TODO + WORKAROUND + HARDCODE, assuming dilation = 1, padding = 0, stride = 1, and with channel_last setting
                _ret_type = RankedTensorType(
                    dimensions=[
                        _lhs_shape[0],
                        Dimension(_lhs_shape[1].value + 1 - _rhs_shape[0].value),
                        Dimension(_lhs_shape[2].value + 1 - _rhs_shape[1].value),
                        _rhs_shape[3],
                    ],
                    element_type=_lhs_type.element_type,
                )

                for _ret_op_element in _ret_op:
                    _ret_sym_entry = SymbolEntry(_ret_op_element.id, _ret_type)
                    global_symbol_table.insert(_ret_sym_entry)

            else:
                assert 0, "found unsupported call method, please check <annotate_type_visitor>"

        else:
            assert 0, "Not support AssignOp with supported RHS value"

        # anchor
        # TODO pattern matching or wrap to utils
        """
        if isinstance(node.value.op, ast.Add):
            assert 0
        elif isinstance(node.value.op, ast.Sub):
            assert 0
        elif isinstance(node.value.op, ast.Mult):
            assert 0
        elif isinstance(node.value.op, ast.Div):
            assert 0
        elif isinstance(node.value.op, ast.Mod):
            assert 0
        elif isinstance(node.value.op, ast.Pow):
            assert 0
        elif isinstance(node.value.op, ast.LShift):
            assert 0
        elif isinstance(node.value.op, ast.RShift):
            assert 0
        elif isinstance(node.value.op, ast.BitOr):
            assert 0
        elif isinstance(node.value.op, ast.BitXor):
            assert 0
        elif isinstance(node.value.op, ast.BitAnd):
            assert 0
        elif isinstance(node.value.op, ast.FloorDiv):
            assert 0
        else:
            assert 0

        """

        super().generic_visit(node)
        return node

    def visit_Return(self, node: ast.AST) -> ast.AST:
        """handle typing annotation and check, infer the operand and return
        types by lookuping symbol table; and create corresponding symbol
        entries into symbol table.

        Args:
            node (ast.AST): AssignOp in python ast.AST.

        Returns:
            ast.AST: same type as input.
        """
        print(self.__str__(), "::visit_Return\n")
        assert isinstance(node.value, ast.Name), "Not handle this type rhs value for AssignOp"
        _argname = node.value.id
        _ret_sym_entry = global_symbol_table.lookup(_argname)
        if _ret_sym_entry is None:
            global_symbol_table.pass_again = True
            super().generic_visit(node)
            return node

        # HANDLE OPERAND SHAPE
        _func_ret_type = _ret_sym_entry.get_type()
        _func_ret_sym_entry = SymbolEntry("ReturnTypeForFunctionDef", _func_ret_type)
        global_symbol_table.insert(_func_ret_sym_entry)

        super().generic_visit(node)
        return node
