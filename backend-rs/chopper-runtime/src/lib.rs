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

use base::constants::*;
use buffer_view::*;
use device_context::*;
use instance::*;
use instruction::*;
use session::*;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn rust_func() -> () {
    let ist = instance::DeviceInstance::new();
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
    println!("runned")
    // TODO package this assert macro into utils, hide rmax_all setting from hardcode
}

#[pymodule]
fn rust(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(rust_func))?;
    Ok(())
}

/*
 * legacy code for bin config
fn main() {
    // entry point for legacy vm mock
    let ist = DeviceInstance::new();
    let mut ipt = interpreter::Interpreter::new(&ist);
    //TODO interactively run, not suitable for test and debug
    //ipt.run();
    //
    // TODO merge session inside
    let status = ipt.mock_operation("%0 = crt.literal.const.i32! 13\n");
    assert_eq!(status.is_ok(), true);
    let status_code = status.unwrap();
    assert_eq!(status_code, 0);
    //
    // entry point for vulkan demo
    //
    /*
    let ist = DeviceInstance::new();
    let mut se = Session::new(&ist);
    se.init();
    let lhs = vec![1.0, 2.0, 3.0];
    let rhs = vec![11.0, 13.0, 17.0];
    let opcode = OpCode::ADD;
    let outs = se.benchmark_run(opcode, lhs, rhs);
    println!("Outputs = {:?}", outs);
    */

    // step 2, use instance info to create device context
    //let mut dc = DeviceContext::new(&ist);
    //dc.load_and_compile_glsl("kernel_add.comp");
    //dc.compute(&ist);

    // step 3, init user session, load and compile kernel code and registry to cache it.
    // Session
    //     init_context
    //         1. init devices
    //         2. init_kernels:: () -> self.KernelRegistry
    //     dispatch<Opcode> -> Functor<Opcode>
    //     run(data: Functor) -> Functor
    //     Functor<Opcode>::compute(data: Buffer) -> Buffer

    // trait hal::Instance<B: Backend>
}
*/
