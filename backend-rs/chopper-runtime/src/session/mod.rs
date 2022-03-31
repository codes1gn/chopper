extern crate backend_vulkan as concrete_backend;
extern crate hal;

use std::{borrow::Cow, env, fs, iter, path::Path, ptr, slice, str::FromStr, sync::Arc};

use hal::prelude::*;
use hal::{adapter::*, buffer, command, memory, pool, prelude::*, pso, query::Type};

use crate::base::kernel::*;
use crate::base::*;
use crate::buffer_view::*;
use crate::device_context::*;
use crate::functor::*;
use crate::instance::*;
use crate::instruction::*;

#[derive(Debug)]
pub(crate) struct Session<'a> {
    pub(crate) device_instance_ref: &'a DeviceInstance,
    pub(crate) device_context: DeviceContext,
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
        // TODO support more kernels
        // TODO get rid of path hardcode by cargo manage datafiles of kernels
        // let kernel_path = vec![kernel::KERNELPATH];
        let path = env::current_dir().unwrap();
        println!("{}", path.display());

        self.device_context.register_kernels(
            "/root/project/glsl_src/binary_arithmetic_f32.comp",
            String::from("binary_arithmetic_f32"),
        );
        self.device_context.register_kernels(
            "/root/project/glsl_src/binary_arithmetic_i32.comp",
            String::from("binary_arithmetic_i32"),
        );
    }

    pub fn benchmark_run<T: SupportedType + std::clone::Clone + std::default::Default>(
        &mut self,
        opcode: OpCode,
        lhs_dataview: DataView<concrete_backend::Backend, T>,
        rhs_dataview: DataView<concrete_backend::Backend, T>,
    ) -> DataView<concrete_backend::Backend, T> {
        self.device_context.device.start_capture();
        let outs = self.run::<T>(opcode, lhs_dataview, rhs_dataview);
        self.device_context.device.stop_capture();
        outs
    }

    pub fn run<T: SupportedType + std::clone::Clone + std::default::Default>(
        &mut self,
        opcode: OpCode,
        lhs_dataview: DataView<concrete_backend::Backend, T>,
        rhs_dataview: DataView<concrete_backend::Backend, T>,
    ) -> DataView<concrete_backend::Backend, T> {
        // step 2 open a physical compute device

        // step 4 load compiled spirv

        let mut functor = Functor::new();
        // TODO move shader in cache
        // TODO add dispatch opcode and dispatch it dynamically later
        // let shader = self.device_context.dispatch_kernel(OpCode::ADDF32);
        let shader = self.device_context.dispatch_kernel(opcode);
        functor.bind(shader);

        // TODO support partial apply for one operator
        let mut result_buffer = functor.apply::<T>(
            &mut self.device_context,
            &self.device_instance_ref,
            lhs_dataview,
            rhs_dataview,
            opcode,
        );
        // update dataview with new value
        result_buffer.eval(&self.device_context.device);
        // print outs
        result_buffer
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_create_session() {
        // defaultly to Add, TODO, add more dispatch path
        let mut dist = DeviceInstance::new();
        let session = DeviceContext::new(&dist);
        assert_eq!(0, 0);
    }

    #[test]
    fn test_e2e_add() {
        let ist = DeviceInstance::new();
        let mut se = Session::new(&ist);
        se.init();
        let lhs = vec![1.0, 2.0, 3.0];
        let rhs = vec![11.0, 13.0, 17.0];
        let lhs_shape = vec![lhs.len()];
        let rhs_shape = vec![lhs.len()];
        // create lhs dataview
        let mut lhs_dataview = DataView::<concrete_backend::Backend, f32>::new(
            &se.device_context.device,
            &se.device_instance_ref.memory_property().memory_types,
            lhs,
            ElementType::F32,
            lhs_shape,
        );
        let mut rhs_dataview = DataView::<concrete_backend::Backend, f32>::new(
            &se.device_context.device,
            &se.device_instance_ref.memory_property().memory_types,
            rhs,
            ElementType::F32,
            rhs_shape,
        );
        let opcode = OpCode::ADDF32;
        let mut result_buffer = se.benchmark_run(opcode, lhs_dataview, rhs_dataview);
        assert_eq!(result_buffer.raw_data, vec!(12.0, 15.0, 20.0));
    }

    #[test]
    fn test_e2e_sub() {
        let ist = DeviceInstance::new();
        let mut se = Session::new(&ist);
        se.init();
        let lhs = vec![1.0, 2.0, 3.0];
        let rhs = vec![11.0, 13.0, 17.0];
        let lhs_shape = vec![lhs.len()];
        let rhs_shape = vec![lhs.len()];
        // create lhs dataview
        let mut lhs_dataview = DataView::<concrete_backend::Backend, f32>::new(
            &se.device_context.device,
            &se.device_instance_ref.memory_property().memory_types,
            lhs,
            ElementType::F32,
            lhs_shape,
        );
        let mut rhs_dataview = DataView::<concrete_backend::Backend, f32>::new(
            &se.device_context.device,
            &se.device_instance_ref.memory_property().memory_types,
            rhs,
            ElementType::F32,
            rhs_shape,
        );
        let opcode = OpCode::SUBF32;
        let mut result_buffer = se.benchmark_run(opcode, lhs_dataview, rhs_dataview);
        assert_eq!(result_buffer.raw_data, vec!(-10.0, -11.0, -14.0));
    }
}
