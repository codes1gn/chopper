extern crate backend_vulkan as concrete_backend;
extern crate hal;

use std::collections::HashMap;

use hal::prelude::*;
use hal::{adapter::*, buffer, command, memory, pool, prelude::*, pso, query::Type};

use crate::base::kernel::*;
use crate::device_context::*;
use crate::instruction::*;

#[derive(Debug)]
pub(crate) struct KernelRegistry {
    executable_cache_table: HashMap<String, KernelByteCode>,
}

impl KernelRegistry {
    pub fn new() -> KernelRegistry {
        return Self {
            executable_cache_table: HashMap::new(),
        };
    }

    fn query_kernel_cache(&self, opcode: OpCode, query_entry: String) -> &KernelByteCode {
        // TODO dummy impl
        return self.executable_cache_table.get(&query_entry).unwrap();
    }

    pub fn register(&mut self, kernel: KernelByteCode, query_entry: String) {
        self.executable_cache_table.insert(query_entry, kernel);
    }

    pub fn dispatch_kernel(&self, dc: &DeviceContext, op: OpCode, query_entry: String) -> Kernel {
        let shader = unsafe {
            dc.device
                .create_shader_module(self.query_kernel_cache(op, query_entry))
        }
        .unwrap();
        shader
    }
}
