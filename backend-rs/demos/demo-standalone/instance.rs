extern crate backend_vulkan as concrete_backend;
extern crate hal;

use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr, sync::Arc};

use hal::prelude::*;
use hal::{adapter::*, buffer, command, memory, pool, prelude::*, pso, query::Type};

// TODO add id and versioning
pub(crate) struct DeviceInstance {
    instance: concrete_backend::Instance,
    memory_property: MemoryProperties,
    queue_family: concrete_backend::QueueFamily,
}

impl DeviceInstance {
    pub(crate) fn new() -> DeviceInstance {
        let _instance = concrete_backend::Instance::create("chopper", 1)
            .expect("failed to create hal backend instance");
        let _adapter = _instance
            .enumerate_adapters()
            .into_iter()
            .find(|adapter| {
                adapter
                    .queue_families
                    .iter()
                    .any(|family| family.queue_type().supports_compute())
            })
            .expect("failed to get computable GPU device");
        let _queue_family = _instance
            .enumerate_adapters()
            .into_iter()
            .find(|adapter| {
                adapter
                    .queue_families
                    .iter()
                    .any(|family| family.queue_type().supports_compute())
            })
            .expect("failed to get computable GPU device")
            .queue_families
            .into_iter()
            .find(|family| family.queue_type().supports_compute())
            .expect("failed to query computable queue_family");
        let _memory_property = _adapter.physical_device.memory_properties();
        return Self {
            instance: _instance,
            memory_property: _memory_property,
            queue_family: _queue_family,
        };
    }

    pub(crate) fn instance(&self) -> &concrete_backend::Instance {
        &self.instance
    }

    pub(crate) fn computable_adapter(&self) -> Adapter<concrete_backend::Backend> {
        self.instance
            .enumerate_adapters()
            .into_iter()
            .find(|adapter| {
                adapter
                    .queue_families
                    .iter()
                    .any(|family| family.queue_type().supports_compute())
            })
            .expect("failed to get computable GPU device")
    }

    pub(crate) fn memory_property(&self) -> &MemoryProperties {
        &self.memory_property
    }

    pub(crate) fn queue_family(&self) -> &concrete_backend::QueueFamily {
        &self.queue_family
    }

    // TODO need to make sure this create method only run once
    pub(crate) fn device_and_queue(&self) -> hal::adapter::Gpu<concrete_backend::Backend> {
        let mut device_and_queue = unsafe {
            self.computable_adapter()
                .physical_device
                .open(&[(&self.queue_family, &[1.0])], hal::Features::empty())
                .unwrap()
        };
        device_and_queue
    }
}

/*
impl Drop for DeviceInstance {
    fn drop(&mut self) {
        unsafe {
            println!("release device context resources");
        };
    }
}
*/

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn create_device_context() {
        let new_dc = DeviceInstance::new();
        assert_eq!(0, 0);
    }
}
