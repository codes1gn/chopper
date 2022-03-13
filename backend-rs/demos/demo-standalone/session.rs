extern crate backend_vulkan as concrete_backend;
extern crate hal;

use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr, sync::Arc};

use hal::prelude::*;
use hal::{adapter::*, buffer, command, memory, pool, prelude::*, pso, query::Type};

use crate::buffer_view::*;
use crate::device_context::*;
use crate::functor::*;
use crate::instance::*;
use crate::kernel::*;
use crate::opcode::*;

pub(crate) struct Session<'a> {
    pub device_instance_ref: &'a DeviceInstance,
    device_context: DeviceContext,
}

impl<'a> Session<'a> {
    pub fn new(dist: &'a DeviceInstance) -> Session {
        let mut device_context = DeviceContext::new(dist);
        return Self {
            device_instance_ref: dist,
            device_context: device_context,
        };
    }

    pub fn init(&mut self) {
        // TODO support more kernels launched at first
        self.device_context
            .register_kernels("./demo-standalone/kernel_add.comp");
    }

    pub fn benchmark_run(&mut self, opcode: OpCode, lhs: Vec<f32>, rhs: Vec<f32>) -> Vec<f32> {
        self.device_context.device.start_capture();
        let outs = self.run(opcode, lhs, rhs);
        self.device_context.device.stop_capture();
        outs
    }

    pub fn run(&mut self, opcode: OpCode, lhs: Vec<f32>, rhs: Vec<f32>) -> Vec<f32> {
        // step 2 open a physical compute device

        // step 4 load compiled spirv

        let mut add_functor = Functor::new();
        let _shader = self.device_context.dispatch_kernel(OpCode::ADD);
        add_functor.bind(_shader);

        // create lhs dataview
        let mut lhs_dataview = DataView::<concrete_backend::Backend>::new(
            &self.device_context.device,
            &self.device_instance_ref.memory_property().memory_types,
            lhs,
        );

        // create rhs dataview
        let mut rhs_dataview = DataView::<concrete_backend::Backend>::new(
            &self.device_context.device,
            &self.device_instance_ref.memory_property().memory_types,
            rhs,
        );

        // TODO support partial apply for one operator
        let mut result_buffer = add_functor.apply(
            &mut self.device_context,
            &self.device_instance_ref,
            lhs_dataview,
            rhs_dataview,
        );
        // print outs
        let outs = result_buffer.val(&self.device_context.device);
        println!("{:?}", outs);
        outs
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn create_session() {
        // defaultly to Add, TODO, add more dispatch path
        let mut dist = DeviceInstance::new();
        let session = DeviceContext::new(&dist);
        assert_eq!(0, 0);
    }
}
