import ast
import torch
import astunparse

from chopper.scaffold.utils import *
from mlir import astnodes
from .node_visitor_base import NodeVisitorBase
from chopper.scaffold.utils.builders import *
from mlir.astnodes import (
    NamedArgument,
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

            if self.arg_annotation[arg_index] is None:
                _argtype = TypeBuilder.create("none")
            else:
                # TODO use From Trait or other way to make this conversion elegant
                if self.arg_annotation[arg_index][1] is torch.float32:
                    _argtype = TypeBuilder.create("tensor", shape=self.arg_annotation[arg_index][0], dtype="f32")
                else:
                    assert 0, "Not support this dtype for annotation"
                # record this type for autodiff use only if it is not None
                func_return_for_autodiff.append(_argtype)

                # record save for backward activations as arguments of bp compute
                _activation_arg = NamedArgument(
                    name=MlirSsaId(value=_argname + "_activation", op_no=None), type=_argtype
                )

                activation_save_for_autodiff.append(_activation_arg)
                # record this type for symbol table use even it is NoneType of self
                ValueBuilder.create(_argname, _argtype)

        ValueBuilder.create("AutodiffFuncReturnType", func_return_for_autodiff)
        ValueBuilder.create("ActivationSaveForAutodiff", activation_save_for_autodiff)

        super().generic_visit(node)
        return node

    def visit_Assign(self, node: ast.AST) -> ast.AST:
        """handle typing annotation and check, infer the operand and return
        types by builders; and create corresponding symbol
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
            _lhs_argname = _rhs_stmt.left.id
            _rhs_argname = _rhs_stmt.right.id
            _opcode = _rhs_stmt.op

            # VERIFY SYMBOL TABLE IF READY FOR THIS OP
            _lhs_type = ValueBuilder.get_type(_lhs_argname)
            _rhs_type = ValueBuilder.get_type(_rhs_argname)
            if _lhs_type is None or _rhs_type is None:
                super().generic_visit(node)
                return node

            # HANDLE OPERAND SHAPE
            assert _lhs_type.element_type == _rhs_type.element_type
            # TODO move this check into OP builder
            # _lhs_shape = _lhs_type.dimensions
            # _rhs_shape = _rhs_type.dimensions
            # assert _lhs_shape == _rhs_shape, "expected same shape of lhs and rhs arguments"
            for _ret_op_element in _ret_op:
                ValueBuilder.create(_ret_op_element.id, _lhs_type)

        elif isinstance(_rhs_stmt, ast.Call):
            print(astunparse.dump(_rhs_stmt))
            # TODO change this way of matching into String Utils
            # e.g. torch.nn.functional.linear and
            # torch.matmul can be explored equally
            # _call_lib = _rhs_stmt.func.value.id  # torch
            # assert _rhs_stmt.func.value.id == "torch", "Found function call other than Torch DSL"

            _call_method = _rhs_stmt.func.attr  # exp or add
            _args = _rhs_stmt.args  # ast.Name, ast.Name
            _arg_type_list = [ValueBuilder.get_type(_argname.id) for _argname in _args]

            # if any arg types are not inferred, means the infer of this call op is not ready
            # run the pass again
            for _ in _arg_type_list:
                if _ is None:
                    super().generic_visit(node)
                    return node

            # handle accepted function calls and assert for guardian
            # TODO build a builder or conversion or mapping utils
            # TODO pattern matching or wrap to utils
            if _call_method == "exp" or _call_method == "tanh":

                assert len(_arg_type_list) == 1, "expected unary, too long of arguments for unaryop call"
                _result_type = _arg_type_list[0]
                for _ret_op_element in _ret_op:
                    ValueBuilder.create(_ret_op_element.id, _result_type)

            elif _call_method == "add" or _call_method == "sub" or _call_method == "mul":

                assert len(_arg_type_list) == 2, "expected binary, too long of arguments for unaryop call"
                _lhs_type = _arg_type_list[0]
                _rhs_type = _arg_type_list[1]
                assert _lhs_type.element_type == _rhs_type.element_type
                assert _lhs_type.dimensions == _rhs_type.dimensions, "expected same shape of lhs and rhs arguments"
                for _ret_op_element in _ret_op:
                    ValueBuilder.create(_ret_op_element.id, _lhs_type)
            elif _call_method == "linear" or _call_method == "matmul":
                assert len(_arg_type_list) == 2, "expected binary, too long of arguments for unaryop call"
                _lhs_type = _arg_type_list[0]
                _rhs_type = _arg_type_list[1]
                assert _lhs_type.element_type == _rhs_type.element_type
                # anchor
                _ret_type = TypeBuilder.create(
                    "tensor", from_lhs_tensor=_lhs_type, from_rhs_tensor=_rhs_type, bin_op="matmul"
                )
                for _ret_op_element in _ret_op:
                    ValueBuilder.create(_ret_op_element.id, _ret_type)
            elif _call_method == "conv2d":
                assert len(_arg_type_list) == 2, "expected binary, too long of arguments for unaryop call"
                _lhs_type = _arg_type_list[0]
                _rhs_type = _arg_type_list[1]
                assert _lhs_type.element_type == _rhs_type.element_type
                # TODO + WORKAROUND + HARDCODE, assuming dilation = 1, padding = 0, stride = 1, and with channel_last setting
                _ret_type = TypeBuilder.create(
                    "tensor", from_lhs_tensor=_lhs_type, from_rhs_tensor=_rhs_type, bin_op="conv-nhwc-hwco"
                )

                for _ret_op_element in _ret_op:
                    ValueBuilder.create(_ret_op_element.id, _ret_type)

            else:
                assert 0, "found unsupported call method, please check <annotate_type_visitor>"

        else:
            assert 0, "Not support AssignOp with supported RHS value"

        super().generic_visit(node)
        return node

    def visit_Return(self, node: ast.AST) -> ast.AST:
        """handle typing annotation and check, infer the operand and return
        types by builders; and create corresponding symbol
        entries into symbol table.

        Args:
            node (ast.AST): AssignOp in python ast.AST.

        Returns:
            ast.AST: same type as input.
        """
        print(self.__str__(), "::visit_Return\n")
        assert isinstance(node.value, ast.Name), "Not handle this type rhs value for AssignOp"
        _argname = node.value.id
        _func_ret_type = ValueBuilder.get_type(_argname)
        if _func_ret_type is None:
            super().generic_visit(node)
            return node

        # HANDLE OPERAND SHAPE
        ValueBuilder.create("ReturnTypeForFunctionDef", _func_ret_type)

        super().generic_visit(node)
        return node
