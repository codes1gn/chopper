from typing import Tuple, List, Optional, Union
import ast
import astunparse
import numpy as np
import logging

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
    ATIR_RandomNormalOp,
    ATIR_RandomUniformOp
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
    def create_replica_merges(cls, operand: astnodes.SsaId) -> List[astnodes.Operation]:
        # ValueBuilder.verbose_symbol_table()
        _replicas = ValueBuilder.get_replicas(operand)
        _operations = []
        if len(_replicas) == 0:
            return _operations
        elif len(_replicas) == 1:
            _operations.append(cls.create_unary("identity", "forward", operand, _replicas[0]))
            return _operations

        assert len(_replicas) > 1
        while len(_replicas) > 1:
            _lhs_operand = _replicas.pop()
            _rhs_operand = _replicas.pop()
            if len(_replicas) == 0:
                _operations.append(cls.create_binary("add", "forward", operand, _lhs_operand, _rhs_operand))
                return _operations
            else:
                _, _ret = cls.create_binary_with_retval("add", "backward", operand, _lhs_operand, _rhs_operand)
                # print(_)
                # print(_ret)
                assert 0
                _replicas.append(_ret)
                _operations.append(_)

    @classmethod
    def create_binary_with_retval(
        cls,
        func: str,
        graph: str,
        retval: astnodes.SsaId,
        lhs_operand: astnodes.SsaId,
        rhs_operand: astnodes.SsaId,
        is_replica: bool = True
    ) -> (astnodes.Operation, astnodes.SsaId):
        if graph == "backward":
            restype = ValueBuilder.get_type(retval.value, mode="backward+savedact")
            lhs_type = ValueBuilder.get_type(lhs_operand.value, mode="backward+savedact")
            rhs_type = ValueBuilder.get_type(rhs_operand.value, mode="backward+savedact")
        elif graph == "forward":
            retval_new = retval
            restype = ValueBuilder.get_type(retval.value, mode="backward")
            lhs_type = ValueBuilder.get_type(lhs_operand.value, mode="backward")
            rhs_type = ValueBuilder.get_type(rhs_operand.value, mode="backward")
        if func == "add":
            if is_replica:
                retval_new = ValueBuilder.get_value_or_replica(retval, restype)
            else:
                retval_new = ValueBuilder.get_value_or_chain(retval, restype)
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_AddOp(
                match=0,
                operand_a=lhs_operand,
                operand_b=rhs_operand,
                dtype=FunctionType(argument_types=[lhs_type, rhs_type], result_types=[restype]),
            )
            return astnodes.Operation(result_list=_result, op=_op, location=None), retval_new
        elif func == "sub":
            if is_replica:
                retval_new = ValueBuilder.get_value_or_replica(retval, restype)
            else:
                retval_new = ValueBuilder.get_value_or_chain(retval, restype)
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_SubOp(
                match=0,
                operand_a=lhs_operand,
                operand_b=rhs_operand,
                dtype=FunctionType(argument_types=[lhs_type, rhs_type], result_types=[restype]),
            )
            return astnodes.Operation(result_list=_result, op=_op, location=None), retval_new
        elif func == "mul":
            if is_replica:
                retval_new = ValueBuilder.get_value_or_replica(retval, restype)
            else:
                retval_new = ValueBuilder.get_value_or_chain(retval, restype)
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_MulOp(
                match=0,
                operand_a=lhs_operand,
                operand_b=rhs_operand,
                dtype=FunctionType(argument_types=[lhs_type, rhs_type], result_types=[restype]),
            )
            return astnodes.Operation(result_list=_result, op=_op, location=None), retval_new
        elif func == "transpose":
            # HARDCODE, how to fetch the const literal and infer the restype?
            restype_transposed = TypeBuilder.create("tensor", from_unary_tensor=restype, transpose_order=[1, 0])
            retval_new = ValueBuilder.get_activation_or_chain(retval, restype)
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_TransposeOp(
                match=0,
                operand_a=lhs_operand,
                operand_b=rhs_operand,
                dtype=FunctionType(argument_types=[lhs_type, rhs_type], result_types=[restype_transposed]),
            )
            # HARDCODE, update table type
            autodiff_saved_activation_table.type_table[retval_new.value] = restype_transposed
            autodiff_symbol_table.type_table[retval_new.value] = restype_transposed

            return astnodes.Operation(result_list=_result, op=_op, location=None), retval_new
        else:
            assert 0

    @classmethod
    def create_binary(
        cls,
        func: str,
        graph: str,
        retval: astnodes.SsaId,
        lhs_operand: astnodes.SsaId,
        rhs_operand: astnodes.SsaId,
    ) -> astnodes.Operation:
        if graph == "backward":
            restype = ValueBuilder.get_type(retval.value, mode=graph)
            retval_new = ValueBuilder.get_value_or_replica(retval, restype)
            lhs_type = ValueBuilder.get_type(lhs_operand.value, mode="backward+savedact")
            rhs_type = ValueBuilder.get_type(rhs_operand.value, mode="backward+savedact")
        elif graph == "forward":
            retval_new = retval
            restype = ValueBuilder.get_type(retval.value, mode="backward")
            lhs_type = ValueBuilder.get_type(lhs_operand.value, mode="backward")
            rhs_type = ValueBuilder.get_type(rhs_operand.value, mode="backward")
        if func == "add":
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_AddOp(
                match=0,
                operand_a=lhs_operand,
                operand_b=rhs_operand,
                dtype=FunctionType(argument_types=[lhs_type, rhs_type], result_types=[restype]),
            )
            return astnodes.Operation(result_list=_result, op=_op, location=None)
        elif func == "sub":
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_SubOp(
                match=0,
                operand_a=lhs_operand,
                operand_b=rhs_operand,
                dtype=FunctionType(argument_types=[lhs_type, rhs_type], result_types=[restype]),
            )
            return astnodes.Operation(result_list=_result, op=_op, location=None)
        elif func == "mul":
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_MulOp(
                match=0,
                operand_a=lhs_operand,
                operand_b=rhs_operand,
                dtype=FunctionType(argument_types=[lhs_type, rhs_type], result_types=[restype]),
            )
            return astnodes.Operation(result_list=_result, op=_op, location=None)
        elif func == "matmul":
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_MatmulOp(
                match=0,
                operand_a=lhs_operand,
                operand_b=rhs_operand,
                dtype=FunctionType(argument_types=[lhs_type, rhs_type], result_types=[restype]),
            )
            return astnodes.Operation(result_list=_result, op=_op, location=None)

        else:
            assert 0

    # TODO if has retval, name it at func signiture
    @classmethod
    def create_const(cls, retval: astnodes.SsaId, literal: Union[List, float, int]) -> (astnodes.Operation, astnodes.SsaId):
        if isinstance(literal, List):
            _forced_type = TypeBuilder.create("tensor", shape=np.array(literal).shape, dtype="i32")
            retval_new = ValueBuilder.get_value_or_chain(retval, _forced_type)
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            # %a_0 = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
            _op = ATIR_ConstShapeOp(
                match=0,
                value=literal,
                dtype=_forced_type,
            )
        else:
            _forced_type = TypeBuilder.create("unit")
            retval_new = ValueBuilder.get_value_or_chain(retval, _forced_type)
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            # %a_0 = "tosa.const"() {value = dense<1.0> : tensor<i32>} : () -> tensor<i32>
            _op = ATIR_ConstOp(
                match=0,
                value=literal,
                dtype=_forced_type,
            )

        return astnodes.Operation(result_list=_result, op=_op, location=None), retval_new

    @classmethod
    def create_random(cls, func: str, graph: str, retval: astnodes.SsaId, **kwargs: astnodes.SsaId) -> astnodes.Operation:
        if graph == "forward":
            restype = ValueBuilder.get_type(retval.value)
            args_type = [ValueBuilder.get_type(kwargs[arg].value) for arg in kwargs]
        elif graph == "backward":
            assert 0, "Not support backward of sample random"
        
        func = func.lower()
        if func == "normal":
            _result = [astnodes.OpResult(value=retval, count=None)]
            _op = ATIR_RandomNormalOp(
                match=0,
                mu=kwargs["operand0"],
                sigma=kwargs["operand1"],
                shape=kwargs["shape"],
                dtype=FunctionType(argument_types=args_type, result_types=[restype])
            )
            return astnodes.Operation(result_list=_result, op = _op, location=None)
        elif func == "uniform":
            _result = [astnodes.OpResult(value=retval, count=None)]
            _op = ATIR_RandomUniformOp(
                match=0,
                minval=kwargs["operand0"],
                maxval=kwargs["operand1"],
                shape=kwargs["shape"],
                dtype=FunctionType(argument_types=args_type, result_types=[restype])
            )
            return astnodes.Operation(result_list=_result, op = _op, location=None)
        else:
            assert 0, "Not support other sampling other than Normal and Uniform Distribution"
    
    @classmethod
    def create_unary_for_random(cls, func: str, retval: astnodes.SsaId, operand: astnodes.SsaId) -> astnodes.Operation:
        _types = ValueBuilder.get_type(operand.value, mode="forward+backward+savedact")
        _result = [astnodes.OpResult(value=retval, count=None)]
        _op = ATIR_IdentityOp(
                match=0,
                operand=operand,
                type=FunctionType(argument_types=[_types], result_types=[_types]),
            )
        return astnodes.Operation(result_list=_result, op=_op, location=None)
    
    
    @classmethod
    def create_unary(cls, func: str, graph: str, retval: astnodes.SsaId, operand: astnodes.SsaId) -> astnodes.Operation:
        if graph == "backward":
            _types = ValueBuilder.get_type(operand.value, mode=graph)
            retval_new = ValueBuilder.get_value_or_replica(retval, _types)
        elif graph == "forward":
            _types = ValueBuilder.get_type(operand.value, mode="backward")
            retval_new = retval
        # TODO simplify it with generics
        if func == "identity":
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_IdentityOp(
                match=0,
                operand=operand,
                type=FunctionType(argument_types=[_types], result_types=[_types]),
            )
            return astnodes.Operation(result_list=_result, op=_op, location=None)
        elif func == "negate":
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_NegateOp(
                match=0,
                operand=operand,
                type=FunctionType(argument_types=[_types], result_types=[_types]),
            )
            return astnodes.Operation(result_list=_result, op=_op, location=None)
        elif func == "exp":
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_ExpOp(
                match=0,
                operand=operand,
                type=FunctionType(argument_types=[_types], result_types=[_types]),
            )
            return astnodes.Operation(result_list=_result, op=_op, location=None)
        elif func == "tanh":
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_TanhOp(
                match=0,
                operand=operand,
                type=FunctionType(argument_types=[_types], result_types=[_types]),
            )
            return astnodes.Operation(result_list=_result, op=_op, location=None)

        else:
            assert 0

    @classmethod
    def create_unary_with_retval(cls, func: str, graph: str, retval: astnodes.SsaId, operand: astnodes.SsaId, is_replica: bool = True, is_operand_act: bool = False) -> (astnodes.Operation, astnodes.SsaId):
        if graph == "backward":
            _types = ValueBuilder.get_type(operand.value, mode="backward+savedact")
            if is_replica:
                if is_operand_act:
                    retval_new = ValueBuilder.get_activation_or_replica(retval, _types)
                else:
                    retval_new = ValueBuilder.get_value_or_replica(retval, _types)
            else:
                if is_operand_act:
                    retval_new = ValueBuilder.get_activation_or_chain(retval, _types)
                else:
                    retval_new = ValueBuilder.get_value_or_chain(retval, _types)
        elif graph == "forward":
            _types = ValueBuilder.get_type(operand.value, mode="backward")
            retval_new = retval
        # TODO simplify it with generics
        if func == "identity":
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_IdentityOp(
                match=0,
                operand=operand,
                type=FunctionType(argument_types=[_types], result_types=[_types]),
            )
            return astnodes.Operation(result_list=_result, op=_op, location=None), retval_new
        elif func == "negate":
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_NegateOp(
                match=0,
                operand=operand,
                type=FunctionType(argument_types=[_types], result_types=[_types]),
            )
            return astnodes.Operation(result_list=_result, op=_op, location=None), retval_new
        elif func == "exp":
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_ExpOp(
                match=0,
                operand=operand,
                type=FunctionType(argument_types=[_types], result_types=[_types]),
            )
            return astnodes.Operation(result_list=_result, op=_op, location=None), retval_new
        elif func == "tanh":
            _result = [astnodes.OpResult(value=retval_new, count=None)]
            _op = ATIR_TanhOp(
                match=0,
                operand=operand,
                type=FunctionType(argument_types=[_types], result_types=[_types]),
            )
            return astnodes.Operation(result_list=_result, op=_op, location=None), retval_new

        else:
            assert 0

    @classmethod
    def create_function(
        cls, func_name: str, arguments: List[astnodes.NamedArgument], restypes: List[astnodes.Type]
    ) -> astnodes.Operation:
        _function = astnodes.Function(
            name=astnodes.SymbolRefId(value=func_name),
            args=arguments,
            result_types=restypes,
            region=astnodes.Region(body=[astnodes.Block(label=None, body=[])]),
            attributes=None,
        )
        return astnodes.Operation(result_list=[], op=_function, location=None)

    @classmethod
    def create_return(
        cls, arguments: List[astnodes.NamedArgument], restypes: List[astnodes.Type]
    ) -> astnodes.Operation:
        _return = ReturnOperation(match=1)
        _return.values = arguments
        _return.types = restypes
        return astnodes.Operation(result_list=None, op=_return, location=None)


class ValueBuilder(object):
    @classmethod
    def get_replicas(cls, value: astnodes.SsaId) -> List[astnodes.SsaId]:
        postfix = 0
        replicas = []
        _value_name = value.value
        while True:
            _trial_name = _value_name + "-{}".format(postfix)
            _trial_value = autodiff_symbol_table.lookup(_trial_name, "value")
            if _trial_value:
                replicas.append(_trial_value)
                postfix += 1
            else:
                return replicas

    @classmethod
    def get_activation_or_replica(cls, value: astnodes.SsaId, value_type: astnodes.Type) -> astnodes.SsaId:
        _value_name = value.value
        # try to create self
        if autodiff_saved_activation_table.lookup(_value_name, "value"):
            postfix = 0
            while True:
                _trial_name = _value_name + "-{}".format(postfix)
                _trial_value = autodiff_saved_activation_table.lookup(_trial_name, "value")
                if not _trial_value:
                    cls.create(_trial_name, value_type, mode="savedact")
                    return autodiff_saved_activation_table.lookup(_trial_name, "value")
                postfix += 1
        else:
            cls.create(_value_name, value_type, mode="savedact")
            return autodiff_saved_activation_table.lookup(_value_name, "value")

    @classmethod
    def get_value_or_replica(cls, value: astnodes.SsaId, value_type: astnodes.Type) -> astnodes.SsaId:
        _value_name = value.value
        # try to create self
        if autodiff_symbol_table.lookup(_value_name, "value"):
            postfix = 0
            while True:
                _trial_name = _value_name + "-{}".format(postfix)
                _trial_value = autodiff_symbol_table.lookup(_trial_name, "value")
                if not _trial_value:
                    cls.create(_trial_name, value_type, mode="backward")
                    return autodiff_symbol_table.lookup(_trial_name, "value")
                postfix += 1
        else:
            cls.create(_value_name, value_type, mode="backward")
            return autodiff_symbol_table.lookup(_value_name, "value")

    @classmethod
    def get_activation_or_chain(cls, value: astnodes.SsaId, value_type: astnodes.Type) -> astnodes.SsaId:
        _value_name = value.value
        # try to create self
        if autodiff_saved_activation_table.lookup(_value_name, "value"):
            postfix = 0
            while True:
                _trial_name = _value_name + "_{}".format(postfix)
                _trial_value = autodiff_saved_activation_table.lookup(_trial_name, "value")
                if not _trial_value:
                    _ = cls.create_raw_activation(_trial_name, value_type)
                    return _
                postfix += 1
        else:
            cls.create(_value_name, value_type, mode="backward", postfix="")
            return autodiff_saved_activation_table.lookup(_value_name, "value")

    @classmethod
    def get_value_or_chain(cls, value: astnodes.SsaId, value_type: astnodes.Type) -> astnodes.SsaId:
        _value_name = value.value
        # try to create self
        if autodiff_symbol_table.lookup(_value_name, "value"):
            postfix = 0
            while True:
                _trial_name = _value_name + "_{}".format(postfix)
                _trial_value = autodiff_symbol_table.lookup(_trial_name, "value")
                if not _trial_value:
                    cls.create(_trial_name, value_type, mode="backward")
                    return autodiff_symbol_table.lookup(_trial_name, "value")
                postfix += 1
        else:
            cls.create(_value_name, value_type, mode="backward")
            return autodiff_symbol_table.lookup(_value_name, "value")

    @classmethod
    def get_saved_activations(cls, mode: str = "value+type"):
        if mode == "value+type":
            return autodiff_saved_activation_table.get_value_with_type_list()
        elif mode == "type":
            return autodiff_saved_activation_table.get_type_list()
        elif mode == "value":
            return autodiff_saved_activation_table.get_value_list()
        else:
            assert 0, "not implement"

    @classmethod
    def get_func_args_autodiff(cls, mode: str = "value+type"):
        if mode == "value+type":
            return autodiff_func_arguments_table.get_value_with_type_list()
        elif mode == "type":
            return autodiff_func_arguments_table.get_type_list()
        elif mode == "value":
            return autodiff_func_arguments_table.get_value_list()
        else:
            assert 0, "not implement"

    @classmethod
    def get_func_rets_autodiff(cls, mode: str = "value+type"):
        # TODO modify into "match string" in mode, to support value+type combination
        if mode == "value+type":
            return autodiff_func_returns_table.get_value_with_type_list()
        elif mode == "type":
            return autodiff_func_returns_table.get_type_list()
        elif mode == "value":
            return autodiff_func_returns_table.get_value_list()
        else:
            assert 0, "not implement"

    @classmethod
    def get_type_or_retry(cls, value_name: str, mode: str = "forward") -> Optional[astnodes.Type]:
        if "forward" in mode:
            _type = feed_forward_symbol_table.lookup(value_name, "type")
            # TODO simplify this pattern
            if _type:
                return _type
        if "backward" in mode:
            _type = autodiff_symbol_table.lookup(value_name, "type")
            if _type:
                return _type
        if "savedact" in mode:
            _type = autodiff_saved_activation_table.lookup(value_name + "-act", "type")
            if _type:
                return _type
        if "funcarg" in mode:
            _type = autodiff_func_arguments_table.lookup(value_name, "type")
            if _type:
                return _type
        if "funcret" in mode:
            _type = autodiff_func_returns_table.lookup(value_name, "type")
            if _type:
                return _type
        if _type is None:
            # by default, run the passes again if this value is not created
            feed_forward_symbol_table.pass_again = True
            errorstr = "\'" + value_name + " \'is not defined"
            assert 0, errorstr
        return _type

    @classmethod
    def get_type(cls, value_name: str, mode: str = "forward") -> Optional[astnodes.Type]:
        if "forward" in mode:
            _type = feed_forward_symbol_table.lookup(value_name, "type")
            if _type:
                return _type
        if "backward" in mode:
            _type = autodiff_symbol_table.lookup(value_name, "type")
            if _type:
                return _type
        if "savedact" in mode:
            _type = autodiff_saved_activation_table.lookup(value_name, "type")
            if _type:
                return _type
        if "funcarg" in mode:
            _type = autodiff_func_arguments_table.lookup(value_name, "type")
            if _type:
                return _type
        if "funcret" in mode:
            _type = autodiff_func_returns_table.lookup(value_name, "type")
            if _type:
                return _type
        assert _type is not None, "value {} not created".format(value_name)
        return _type

    @classmethod
    def get_value_with_type(cls, value_name: str, mode: str = "forward") -> Optional[astnodes.Type]:
        if "forward" in mode:
            _value = feed_forward_symbol_table.lookup(value_name, "value")
            _type = feed_forward_symbol_table.lookup(value_name, "type")
            if _value is not None and _type is not None:
                return NamedArgument(name=_value, type=_type)
        if "backward" in mode:
            _value = autodiff_symbol_table.lookup(value_name, "value")
            _type = autodiff_symbol_table.lookup(value_name, "type")
            if _value is not None and _type is not None:
                return NamedArgument(name=_value, type=_type)
        if "savedact" in mode:
            _value = autodiff_saved_activation_table.lookup(value_name, "value")
            _type = autodiff_saved_activation_table.lookup(value_name, "type")
            if _value is not None and _type is not None:
                return NamedArgument(name=_value, type=_type)
        if "funcarg" in mode:
            _value = autodiff_func_arguments_table.lookup(value_name, "value")
            _type = autodiff_func_arguments_table.lookup(value_name, "type")
            if _value is not None and _type is not None:
                return NamedArgument(name=_value, type=_type)
        if "funcret" in mode:
            _value = autodiff_func_returns_table.lookup(value_name, "value")
            _type = autodiff_func_returns_table.lookup(value_name, "type")
            if _value is not None and _type is not None:
                return NamedArgument(name=_value, type=_type)
        assert _value is not None and _type is not None
        return NamedArgument(name=_value, type=_type)

    @classmethod
    def get_value(cls, value_name: str, mode: str = "forward") -> Optional[astnodes.Type]:
        # print("querying ==> {}, with mode = {}".format(value_name, mode))
        if "forward" in mode:
            _value = feed_forward_symbol_table.lookup(value_name, "value")
            if _value:
                return _value
        if "backward" in mode:
            _value = autodiff_symbol_table.lookup(value_name, "value")
            if _value:
                return _value
        if "savedact" in mode:
            _value = autodiff_saved_activation_table.lookup(value_name, "value")
            if _value:
                return _value
        if "funcarg" in mode:
            _value = autodiff_func_arguments_table.lookup(value_name, "value")
            if _value:
                return _value
        if "funcret" in mode:
            _value = autodiff_func_returns_table.lookup(value_name, "value")
            if _value:
                return _value
        assert _value is not None
        return _value

    @classmethod
    def create_raw_activation(cls, value_name: str, value_type: astnodes.Type) -> astnodes.SsaId:
        autodiff_symbol_table.insert(value_name, value_type)
        return autodiff_symbol_table.lookup(value_name, "value")

    # forward | backward | savedact | funcarg | funcret
    @classmethod
    def create(cls, value_name: str, value_type: astnodes.Type, mode: str = "forward+backward", postfix="-act"):
        if "forward" in mode:
            if feed_forward_symbol_table.lookup(value_name, "type"):
                assert 0, "error: redefine of value {} with newtype = {}".format(value_name, value_type)
            feed_forward_symbol_table.insert(value_name, value_type)
            print(f"ValueBuilder.create forward symbol: name = { value_name}, type = {value_type}")
        if "backward" in mode:
            if autodiff_symbol_table.lookup(value_name, "type"):
                assert 0, "error: redefine of value {} with, newtype = {}".format(value_name, value_type)
            autodiff_symbol_table.insert(value_name, value_type)
            print(f"ValueBuilder.create backward symbol: name = { value_name}, type = {value_type}")
        if "savedact" in mode:
            if autodiff_saved_activation_table.lookup(value_name + postfix, "type"):
                print("warning: redefine of value {} with, newtype = {}".format(value_name, value_type))
                return
            autodiff_saved_activation_table.insert(value_name + "-act", value_type)
            print(f"ValueBuilder.create savedact symbol: name = { value_name}, type = {value_type}")
        if "funcarg" in mode:
            if autodiff_func_arguments_table.lookup(value_name, "type"):
                assert 0, "error: redefine of value {} with, newtype = {}".format(value_name, value_type)
            autodiff_func_arguments_table.insert(value_name, value_type)
            print(f"ValueBuilder.create funarg symbol: name = { value_name}, type = {value_type}")
        if "funcret" in mode:
            if autodiff_func_returns_table.lookup(value_name, "type"):
                assert 0, "error: redefine of value {} with, newtype = {}".format(value_name, value_type)
            autodiff_func_returns_table.insert(value_name, value_type)
            print(f"ValueBuilder.create funret symbol: name = { value_name}, type = {value_type}")
        return

    @classmethod
    def verbose_symbol_table(cls):
        print(feed_forward_symbol_table)
        print(autodiff_symbol_table)
        print(autodiff_saved_activation_table)
        print(autodiff_func_arguments_table)
        print(autodiff_func_returns_table)


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
        print(cls.__name__, " TypeBuilder::create kwattr is: ", kwattr)
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
