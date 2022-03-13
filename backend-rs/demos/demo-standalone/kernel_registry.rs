extern crate backend_vulkan as concrete_backend;
extern crate hal;

use hal::prelude::*;
use hal::{adapter::*, buffer, command, memory, pool, prelude::*, pso, query::Type};

use crate::device_context::*;
use crate::kernel::*;
use crate::opcode::*;

pub(crate) struct KernelRegistry {
    executable_cache_table: Vec<KernelByteCode>,
}

impl KernelRegistry {
    pub fn new() -> KernelRegistry {
        return Self {
            executable_cache_table: vec![],
        };
    }

    fn query_kernel_cache(&self, opcode: OpCode) -> &KernelByteCode {
        // TODO dummy impl
        return &self.executable_cache_table[0];
    }

    pub fn register(&mut self, kernel: KernelByteCode) {
        self.executable_cache_table.push(kernel);
    }

    pub fn dispatch_kernel(&self, dc: &DeviceContext, op: OpCode) -> Kernel {
        let shader =
            unsafe { dc.device.create_shader_module(self.query_kernel_cache(op)) }.unwrap();
        shader
    }
}
