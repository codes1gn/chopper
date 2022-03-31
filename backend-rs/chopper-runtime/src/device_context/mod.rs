extern crate backend_vulkan as concrete_backend;
extern crate hal;

use std::{borrow::Cow, collections::HashMap, fs, iter, ptr, slice, str::FromStr, sync::Arc};

use hal::prelude::*;
use hal::{adapter::*, buffer, command, memory, pool, prelude::*, pso, query::Type};

use crate::base::kernel::*;
use crate::buffer_view::*;
use crate::instance::*;
use crate::instruction::*;
use crate::kernel::kernel_registry::*;

#[derive(Debug)]
pub(crate) struct DeviceContext {
    // TODO refactor into kernel_registry
    kernel_registry: KernelRegistry,
    //adapter: Adapter<concrete_backend::Backend>,
    //physical_device: concrete_backend::PhysicalDevice,
    //pub device_and_queue: hal::adapter::Gpu<concrete_backend::Backend>,
    pub device: concrete_backend::Device,
    pub queue_groups: Vec<hal::queue::family::QueueGroup<concrete_backend::Backend>>,
    // TODO .first_mut().unwrap(); before use, owner is Functor
    //
    pub descriptor_pool: concrete_backend::native::DescriptorPool,
}

impl DeviceContext {
    pub fn new(di: &DeviceInstance) -> DeviceContext {
        let mut device_and_queue = di.device_and_queue();
        let mut descriptor_pool = unsafe {
            device_and_queue.device.create_descriptor_pool(
                100, // TODO count of desc sets which below max_sets
                iter::once(pso::DescriptorRangeDesc {
                    ty: pso::DescriptorType::Buffer {
                        ty: pso::BufferDescriptorType::Storage { read_only: false },
                        format: pso::BufferDescriptorFormat::Structured {
                            dynamic_offset: false,
                        },
                    },
                    count: 1,
                }),
                pso::DescriptorPoolCreateFlags::empty(),
            )
        }
        .expect("Can't create descriptor pool");
        return Self {
            kernel_registry: KernelRegistry::new(),
            //device_and_queue: device_and_queue,
            device: device_and_queue.device,
            queue_groups: device_and_queue.queue_groups,
            descriptor_pool: descriptor_pool,
        };
    }

    pub fn register_kernels(&mut self, file_path: &str, query_entry: String) {
        // glsl_to_spirv, TODO, support more, spv format and readable spirv ir.
        // TODO, read external config of all kernels, and cache it by OpCode
        // println!("{:?}", file_path);
        let glsl = fs::read_to_string(file_path).unwrap();
        // println!("{:?}", glsl);
        let spirv_file = glsl_to_spirv::compile(&glsl, glsl_to_spirv::ShaderType::Compute).unwrap();
        // println!("{:?}", spirv_file);
        // TODO need to impl implicit conversion
        let spirv: KernelByteCode = auxil::read_spirv(spirv_file).unwrap() as KernelByteCode;
        //let spirv: KernelByteCode = auxil::read_spirv(spirv_file).unwrap() as KernelByteCode;
        self.kernel_registry.register(spirv, query_entry);
    }

    pub fn kernel_registry(&self) -> &KernelRegistry {
        &self.kernel_registry
    }

    pub fn dispatch_kernel(&self, op: OpCode) -> Kernel {
        let query_entry: String = op.to_kernel_query_entry();
        self.kernel_registry.dispatch_kernel(self, op, query_entry)
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn create_add_functor() {
        // defaultly to Add, TODO, add more dispatch path
        let mut ist = DeviceInstance::new();
        let add_functor = DeviceContext::new(&ist);
        assert_eq!(0, 0);
    }
}
