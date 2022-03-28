from typing import Tuple, List, Optional
import ast
import astunparse

from mlir import astnodes
from mlir.astnodes import (
    CustomOperation,
    FunctionType,
    NamedArgument,
    Dimension,
    RankedTensorType,
    DenseElementsAttr,
)
from mlir.dialects.standard import ReturnOperation, ConstantOperation
from chopper.scaffold.mlir_dialects.dialect_tcf import TCF_AddOp, TCF_ExpOp
from chopper.scaffold.mlir_dialects.dialect_atir import (
    ATIR_ConstOp,
    ATIR_IdentityOp,
    ATIR_NegateOp,
    ATIR_AddOp,
    ATIR_SubOp,
    ATIR_MulOp,
    ATIR_ExpOp,
    ATIR_TanhOp,
    ATIR_MatmulOp,
    ATIR_Conv2DChannelFirstOp,
    ATIR_ConstShapeOp,
    ATIR_TransposeOp,
    UnitTensorType,
)


from chopper.pass_manager.symbol_table import *

__all__ = [
    "OpBuilder",
    "ValueBuilder",
    "TypeBuilder",
]


# TODO change str matchers into enum types for safety
class OpBuilder(object):
    @classmethod
    def create(cls, shape: List[int], elem_ty: astnodes.Type):
        return


class ValueBuilder(object):
    @classmethod
    def get_type(cls, value_name: str, expect_exist: bool = False) -> Optional[astnodes.Type]:
        _type = feed_forward_symbol_table.lookup(value_name, "type")
        if not expect_exist:
            if _type is None:
                # by default, run the passes again if this value is not created
                feed_forward_symbol_table.pass_again = True
        else:
            if _type is None:
                assert 0, "value {} not created".format(value_name)

        return _type

    @classmethod
    def get_value(cls, value_name: str) -> Optional[astnodes.Type]:
        _value = feed_forward_symbol_table.lookup(value_name, "value")
        if _value is None:
            assert 0, "value {} not created".format(value_name)

        return _value

    @classmethod
    def create(cls, value_name: str, value_type: astnodes.Type, mode: str = "forward+backward"):
        if feed_forward_symbol_table.lookup(value_name, "type"):
            assert 0, "error: redefine of value {} with oldtype = {}, newtype = {}".format(
                value_name, feed_forward_symbol_table.lookup(value_name, "type"), value_type
            )
        if mode == "forward+backward" or mode == "forward-only":
            feed_forward_symbol_table.insert(value_name, value_type)
        if mode == "forward+backward" or mode == "backward-only":
            autodiff_symbol_table.insert(value_name, value_type)
        return


class TypeBuilder(object):
    @classmethod
    def build_ranked_tensor(
        cls,
        shape: Optional[List[int]] = None,
        dtype: Optional[str] = None,
        from_unary_tensor: Optional[astnodes.RankedTensorType] = None,
        from_lhs_tensor: Optional[astnodes.RankedTensorType] = None,
        from_rhs_tensor: Optional[astnodes.RankedTensorType] = None,
        transpose_order: Optional[List[int]] = None,
        bin_op: Optional[str] = None,
    ) -> astnodes.Type:
        if from_unary_tensor:
            old_dims = from_unary_tensor.dimensions
            old_dtype = from_unary_tensor.element_type
            if transpose_order:
                old_dims = from_unary_tensor.dimensions
                new_dims = [old_dims[new_idx] for new_idx in transpose_order]
                new_dtype = old_dtype
            else:
                new_dims = old_dims
                new_dtype = old_dtype
            return RankedTensorType(
                dimensions=new_dims,
                element_type=new_dtype,
            )
        elif from_unary_tensor is None and (from_lhs_tensor is not None and from_rhs_tensor is not None):
            lhs_dims = from_lhs_tensor.dimensions
            rhs_dims = from_rhs_tensor.dimensions
            assert bin_op is not None
            if bin_op == "matmul":
                new_dims = [lhs_dims[0], rhs_dims[1]]
                return RankedTensorType(
                    dimensions=new_dims,
                    element_type=from_lhs_tensor.element_type,
                )
            elif bin_op == "conv-nhwc-hwco":
                new_dims = [
                    lhs_dims[0],
                    Dimension(lhs_dims[1].value - rhs_dims[0].value + 1),
                    Dimension(lhs_dims[2].value - rhs_dims[1].value + 1),
                    rhs_dims[3],
                ]
                return RankedTensorType(
                    dimensions=new_dims,
                    element_type=from_lhs_tensor.element_type,
                )
            else:
                assert 0, "unimplemented bin-op for tensor type inference"
        else:
            assert shape is not None
            assert dtype is not None
            _dims = [Dimension(dim_idx) for dim_idx in shape]
            _dtype = cls.build_numeric(dtype=dtype)
            return RankedTensorType(
                dimensions=_dims,
                element_type=_dtype,
            )

    @classmethod
    def build_none(cls) -> astnodes.Type:
        return astnodes.NoneType()

    @classmethod
    def build_numeric(cls, dtype: Optional[str] = None) -> astnodes.Type:
        assert dtype is not None
        if dtype == "f32":
            return astnodes.FloatType(astnodes.FloatTypeEnum.f32)
        elif dtype == "f64":
            return astnodes.FloatType(astnodes.FloatTypeEnum.f64)
        elif dtype == "i32":
            return astnodes.SignlessIntegerType(width=32)
        else:
            assert 0, "unknown bitwidth of float type"

    @classmethod
    def create(cls, op: str, **kwattr) -> astnodes.Type:
        print(kwattr)
        if op == "none":
            return cls.build_none()
        elif op == "numeric":
            return cls.build_numeric(**kwattr)
        elif op == "tensor":
            return cls.build_ranked_tensor(**kwattr)
        elif op == "unit":
            # sidepath for unit tensor type
            return UnitTensorType(element_type=cls.build_numeric("f32"))
        else:
            assert 0, "unknown type specified"
