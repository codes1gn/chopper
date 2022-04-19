#[macro_use]
extern crate nom;

extern crate backend_vulkan as vk_types;
extern crate hal;

extern crate transpose;

use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr};

use hal::prelude::*;
use hal::{adapter::Adapter, adapter::MemoryType, buffer, command, memory, pool, prelude::*, pso};

pub mod assembler;
pub mod base;
pub mod instruction;
pub mod interpreter;
pub mod vm;

//pub mod base;
pub mod buffer_view;
pub mod device_context;
pub mod functor;
pub mod instance;
pub mod kernel;
pub mod session;

use instance::*;
use interpreter::*;

use base::constants::*;
use buffer_view::*;
use device_context::*;
use instruction::*;
use session::*;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::ndarray::{array, ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{ToPyArray, IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArray2, PyArray1};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use pyo3::types::{PyDict, PyTuple};

use transpose::*;



#[pyclass(name = "CallableModule")]
pub struct CallableModule {
    // We use `#[pyo3(get)]` so that python can read the count but not mutate it.
    #[pyo3(get)]
    bytecodes: String,
    kernel_option: String,
}

// TODO hardcoded with explictiy PyArray2 types, consider PyTuple or other way to accept variadic
fn get_workaround_forward_bytecodes(kernel_option: &str) -> Vec<String> {
    let mut bc_array: Vec<String> = vec![];

    // TODO (now use manual way to load arg0 and arg1) load arguments for function at %0, %1 .. %arg_cnt
    // bc_array.push(format!("%0 = crt.literal.const.tensor! dense<[0.0], shape=[1]>\n"));
    // bc_array.push(format!("%1 = crt.literal.const.tensor! dense<[0.0], shape=[1]>\n"));

    match kernel_option {
        "add" => {
            // add forward
            bc_array.push(format!("%2 = crt.add.f32! %0, %1 : f32\n"));
        },
        "matmul" => {
            // matmul forward
            bc_array.push(format!("%2 = crt.matmul.f32! %0, %1 : f32\n"));
        },
        _ => {}
    }

    bc_array
}

// TODO hardcoded with explictiy PyArray2 types, consider PyTuple or other way to accept variadic
fn get_workaround_backward_bytecodes(kernel_option: &str) -> Vec<String> {
    let mut bc_array: Vec<String> = vec![];

    // TODO (now use manual way to load arg0 and arg1) load arguments for function at %0, %1 .. %arg_cnt
    // bc_array.push(format!("%0 = crt.literal.const.tensor! dense<[0.0], shape=[1]>\n"));
    // bc_array.push(format!("%1 = crt.literal.const.tensor! dense<[0.0], shape=[1]>\n"));
    //
    match kernel_option {
        "add" => {
            // add forward
            // TODO make a unconsuming functor, 9 should be 0, reusage of register 0
            bc_array.push(format!("%3 = crt.add.f32! %9, %2 : f32\n"));
            bc_array.push(format!("%4 = crt.add.f32! %1, %0 : f32\n"));
        },
        "matmul" => {
            // matmul forward
            // TODO make a unconsuming functor
            // TODO use tmp solution to transpose, so no need to transpose
            // bc_array.push(format!("%6 = crt.tensor.transpose! %2, [1, 0] : f32\n"));
            // bc_array.push(format!("%5 = crt.tensor.transpose! %1, [1, 0] : f32\n"));
            // bc_array.push(format!("%3 = crt.matmul.f32! %9, %6 : f32\n"));
            // bc_array.push(format!("%4 = crt.matmul.f32! %5, %0 : f32\n"));
            bc_array.push(format!("%3 = crt.matmul.f32! %9, %2 : f32\n"));
            bc_array.push(format!("%4 = crt.matmul.f32! %1, %0 : f32\n"));
        },
        _ => {}
    }

    bc_array
}

#[pymethods]
impl CallableModule {
    // Note that we don't validate whether `wraps` is actually callable.
    //
    // While we could use `PyAny::is_callable` for that, it has some flaws:
    //    1. It doesn't guarantee the object can actually be called successfully
    //    2. We still need to handle any exceptions that the function might raise
    #[new]
    fn __new__(bytecodes: String, kernel_option: String) -> Self {
        let kernel = &kernel_option[..];
        match kernel {
            "" => {
                CallableModule { bytecodes: bytecodes, kernel_option: "add".to_string() }
            }
            _ => {
            CallableModule { bytecodes: bytecodes, kernel_option: kernel_option }
            }
        }
    }

    // TODO hardcoded with explictiy PyArray2 types, consider PyTuple or other way to accept
    // variadic
    // may need serde for se and des
    #[args(args = "*", kwargs = "**")]
    fn forward<'py>(
        &mut self,
        py: Python<'py>,
        ist: &DeviceInstance,
        arg0: &PyArray2<f32>,
        arg1: &PyArray2<f32>,
        kwargs: Option<&PyDict>,
    ) -> &'py PyArray1<f32> {

        // println!("create interpreter");
        let mut ipt = interpreter::Interpreter::new(&ist);

        let lhs_operand = 0;
        let rhs_operand = 1;
        let outs_register = 2;

        // parsing args and get func arguments and its shapes
        // TODO change vec to array abstraction on databuffer
        let shape0 = arg0.shape();
        let data0 = unsafe { arg0.as_slice().unwrap() };
        let shape1 = arg1.shape();
        let data1 = unsafe { arg1.as_slice().unwrap() };
        // println!("{:?}", data1);
        // TODO tmp HARDCODE for demo use
        ipt.vm.push_tensor_buffer(lhs_operand, data0.to_vec(), shape0.to_vec());
        ipt.vm.push_tensor_buffer(rhs_operand, data1.to_vec(), shape1.to_vec());

        // let bytecode_array = get_workaround_forward_bytecodes("add");
        let bytecode_array = get_workaround_forward_bytecodes(&self.kernel_option[..]);
        for _bytecode_string in bytecode_array {
            // println!("Executing {}", _bytecode_string.as_str());
            let status = ipt.run_bytecode(_bytecode_string);
        }


        let outs_dataview = ipt.vm.data_buffer_f32.remove(&outs_register).unwrap();
        outs_dataview.raw_data.to_pyarray(py)
        //let _data = vec![
        //    outs_dataview.raw_data[0..3],
        //    outs_dataview.raw_data[3..6],
        //    outs_dataview.raw_data[6..9],
        //];
        //_data.to_pyarray(py)
        //
        // let gil = pyo3::Python::acquire_gil();
        //let outs: &PyArray2<f32> = PyArray2::<f32>::from_vec2(gil.python(), &outs_dataview.raw_data).unwrap();
        // let outs: &PyArray<f32> = PyArray::<f32>::from_vec(gil.python(), &outs_dataview.raw_data).unwrap();
        // outs
    }

    #[args(args = "*", kwargs = "**")]
    fn backward<'py>(
        &mut self,
        py: Python<'py>,
        ist: &DeviceInstance,
        arg0: &PyArray2<f32>,
        arg1: &PyArray2<f32>,
        arg2: &PyArray2<f32>,
        kwargs: Option<&PyDict>,
    ) -> (&'py PyArray1<f32>, &'py PyArray1<f32>) {
        // println!("create interpreter");
        let mut ipt = interpreter::Interpreter::new(&ist);

        let grad = 0;
        let gradworkaroundduplicate = 9;
        let act0 = 1;
        let act1 = 2;
        let lhs_grad = 3;
        let rhs_grad = 4;

        // parsing args and get func arguments and its shapes
        // TODO change vec to array abstraction on databuffer
        let shape0 = arg0.shape();
        let data0 = unsafe { arg0.as_slice().unwrap() };
        let shape1 = arg1.shape();
        let data1 = unsafe { arg1.as_slice().unwrap() };
        let shape2 = arg2.shape();
        let data2 = unsafe { arg2.as_slice().unwrap() };

        let mut data1_transposed = vec![0.0; data1.len()];
        transpose::transpose(&data1.to_vec(), &mut data1_transposed, shape1[0], shape1[1]);
        let shape1_transposed = vec![shape1[1], shape1[0]];
        let mut data2_transposed = vec![0.0; data2.len()];
        transpose::transpose(&data2.to_vec(), &mut data2_transposed, shape2[0], shape2[1]);
        let shape2_transposed = vec![shape2[1], shape2[0]];
        //println!("{:?}", data1);
        //println!("{:?}", data1_transposed);
        //assert_eq!(0, 1);

        // TODO tmp HARDCODE for demo use
        ipt.vm.push_tensor_buffer(grad, data0.to_vec(), shape0.to_vec());
        ipt.vm.push_tensor_buffer(gradworkaroundduplicate, data0.to_vec(), shape0.to_vec());
        ipt.vm.push_tensor_buffer(act0, data1_transposed, shape1_transposed);
        ipt.vm.push_tensor_buffer(act1, data2_transposed, shape2_transposed);
        // TODO use tmp solution to transpose
        //ipt.vm.push_tensor_buffer(act0, data1.to_vec(), shape1.to_vec());
        //ipt.vm.push_tensor_buffer(act1, data2.to_vec(), shape2.to_vec());

        let bytecode_array = get_workaround_backward_bytecodes(&self.kernel_option[..]);
        for _bytecode_string in bytecode_array {
            // println!("Executing {}", _bytecode_string.as_str());
            let status = ipt.run_bytecode(_bytecode_string);
        }


        let lhs_outs = ipt.vm.data_buffer_f32.remove(&lhs_grad).unwrap();
        let rhs_outs = ipt.vm.data_buffer_f32.remove(&rhs_grad).unwrap();
        (lhs_outs.raw_data.to_pyarray(py), rhs_outs.raw_data.to_pyarray(py))
    }

}

#[pymodule]
fn Runtime(py: Python, m: &PyModule) -> PyResult<()> {

    m.add_class::<DeviceInstance>()?;
    m.add_class::<CallableModule>()?;

    #[pyfn(m)]
    fn demo(ist: &DeviceInstance) -> usize {
        // let ist = instance::DeviceInstance::new();
        let mut ipt = interpreter::Interpreter::new(&ist);
        // ok
        let status = ipt.mock_operation("%8 = crt.literal.const.f32! 1.3 : f32\n");
        let status = ipt.mock_operation("%7 = crt.literal.const.f32! 2.9 : f32\n");
        let status = ipt.mock_operation("%1 = crt.literal.const.f32! 7.4 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // add
        let status = ipt.mock_operation("%4 = crt.add.f32! %8, %7 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);

        // sub
        let status = ipt.mock_operation("%5 = crt.sub.f32! %1, %4 : f32\n");
        assert_eq!(status.is_ok(), true);
        let status_code = status.unwrap();
        assert_eq!(status_code, 0);
        // TODO package this assert macro into utils, hide rmax_all setting from hardcode
        31415926
    }

    #[pyfn(m)]
    fn testing(operand: &PyArray2<f32>) -> Vec<f32> {
        //println!("{}", operand);
        vec![1.1]
    }

    #[pyfn(m)]
    fn load_and_invoke(ist: &DeviceInstance, lhs_operand: Vec<f32>, lhs_shape: Vec<usize>, rhs_operand: Vec<f32>, rhs_shape: Vec<usize>, bytecodes: &str) -> Vec<f32> {
        // TODO, accept string of bytecodes, list<f32> or operands, return list<f32>
        // TODO, maybe wrap it into callables
        //println!("lhs-operand = {:?}", lhs_operand);
        //println!("rhs-operand = {:?}", rhs_operand);
        //println!("bytecode instruction = {:?}", bytecodes);

        let lhs_register = 0;
        let rhs_register = 1;
        let outs_register = 2;

        // create arguments
        let mut ipt = interpreter::Interpreter::new(&ist);
        let status = ipt.run_bytecode(format!("%{} = crt.literal.const.tensor! dense<[1.1 2.2 3.3 4.4 5.5 6.6], shape=[2 3]>\n", lhs_register));
        let status = ipt.run_bytecode(format!("%{} = crt.literal.const.tensor! dense<[1.1 2.2 3.3 4.4 5.5 6.6], shape=[2 3]>\n", rhs_register));
        let status = ipt.run_bytecode(format!("%{} = crt.add.f32! %0, %1 : f32\n", outs_register));
        let outs_dataview = ipt.vm.data_buffer_f32.remove(&outs_register).unwrap();
        outs_dataview.raw_data
    }

    Ok(())
}

/*
 * add forward callable: legacy code
    // TODO hardcoded with explictiy PyArray2 types, consider PyTuple or other way to accept
    // variadic
    // may need serde for se and des
    #[args(args = "*", kwargs = "**")]
    fn forward<'py>(
        &mut self,
        py: Python<'py>,
        ist: &DeviceInstance,
        arg0: &PyArray2<f32>,
        arg1: &PyArray2<f32>,
        kwargs: Option<&PyDict>,
    ) -> &'py PyArray1<f32> {

        println!("Initializing Vulkan Device");
        let mut ipt = interpreter::Interpreter::new(&ist);
        println!("Initialized Vulkan Device");

        let lhs_operand = 0;
        let rhs_operand = 1;
        let outs_register = 2;

        // parsing args and get func arguments and its shapes
        // TODO change vec to array abstraction on databuffer
        let shape0 = arg0.shape();
        let data0 = unsafe { arg0.as_slice().unwrap() };
        let shape1 = arg1.shape();
        let data1 = unsafe { arg1.as_slice().unwrap() };
        println!("{:?}", data1);
        // TODO tmp HARDCODE for demo use
        ipt.vm.push_tensor_buffer(lhs_operand, data0.to_vec(), shape0.to_vec());
        ipt.vm.push_tensor_buffer(rhs_operand, data1.to_vec(), shape1.to_vec());

        let bytecode_array = get_workaround_forward_bytecodes(arg0, arg1);
        for _bytecode_string in bytecode_array {
            println!("Executing {}", _bytecode_string.as_str());
            let status = ipt.run_bytecode(_bytecode_string);
        }


        let outs_dataview = ipt.vm.data_buffer_f32.remove(&outs_register).unwrap();
        outs_dataview.raw_data.to_pyarray(py)
        //let _data = vec![
        //    outs_dataview.raw_data[0..3],
        //    outs_dataview.raw_data[3..6],
        //    outs_dataview.raw_data[6..9],
        //];
        //_data.to_pyarray(py)
        //
        // let gil = pyo3::Python::acquire_gil();
        //let outs: &PyArray2<f32> = PyArray2::<f32>::from_vec2(gil.python(), &outs_dataview.raw_data).unwrap();
        // let outs: &PyArray<f32> = PyArray::<f32>::from_vec(gil.python(), &outs_dataview.raw_data).unwrap();
        // outs
    }
 * */
