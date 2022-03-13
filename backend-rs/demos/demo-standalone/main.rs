extern crate backend_vulkan as vk_types;
extern crate hal;

use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr};

use hal::prelude::*;
use hal::{adapter::Adapter, adapter::MemoryType, buffer, command, memory, pool, prelude::*, pso};

pub mod base;
pub mod buffer_view;
pub mod device_context;
pub mod functor;
pub mod instance;
pub mod kernel;
pub mod kernel_registry;
pub mod opcode;
pub mod session;

use base::*;
use buffer_view::*;
use device_context::*;
use instance::*;
use opcode::*;
use session::*;

fn main() {
    // step 1, init device instance, also in VM instance init part
    let ist = DeviceInstance::new();
    let mut se = Session::new(&ist);
    se.init();
    let lhs = vec![1.0, 2.0, 3.0];
    let rhs = vec![11.0, 13.0, 17.0];
    let opcode = OpCode::ADD;
    let outs = se.benchmark_run(opcode, lhs, rhs);
    println!("Outputs = {:?}", outs);

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
