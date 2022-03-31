#[macro_use]
extern crate nom;

extern crate backend_vulkan as vk_types;
extern crate hal;

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
use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArray2};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pymodule]
fn Runtime(py: Python, m: &PyModule) -> PyResult<()> {

    m.add_class::<DeviceInstance>()?;

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
        println!("{}", operand);
        vec![1.1]
    }

    #[pyfn(m)]
    fn load_and_invoke(ist: &DeviceInstance, lhs_operand: Vec<f32>, lhs_shape: Vec<usize>, rhs_operand: Vec<f32>, rhs_shape: Vec<usize>, bytecodes: &str) -> Vec<f32> {
        // TODO, accept string of bytecodes, list<f32> or operands, return list<f32>
        // TODO, maybe wrap it into callables
        println!("lhs-operand = {:?}", lhs_operand);
        println!("rhs-operand = {:?}", rhs_operand);
        println!("bytecode instruction = {:?}", bytecodes);

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

