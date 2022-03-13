extern crate backend_vulkan as concrete_backend;
extern crate hal;

use std::{borrow::Cow, fs, iter, ptr, slice, str::FromStr, sync::Arc};

use hal::prelude::*;
use hal::{adapter::*, buffer, command, memory, pool, prelude::*, pso, query::Type};

use crate::base::constants::*;
use crate::device_context::*;

use crate::base::*;

#[derive(Debug)]
pub(crate) enum BufferType {
    HOST,
    DEVICE,
}

// Buffer Functor of a collection of a BufferRef at differing memory hierachy
#[derive(Debug)]
pub(crate) struct DataView<B: hal::Backend, T> {
    pub host_buffer: BufferView<B>,
    pub device_buffer: BufferView<B>,
    pub raw_data: Vec<T>,
    pub data_size: usize,
}

impl<B: hal::Backend, T> DataView<B, T> {
    pub fn new(
        device: &B::Device,
        memory_types: &[MemoryType],
        data: Vec<T>,
        dtype: ElementType,
    ) -> DataView<B, T> {
        // TODO tobe handled by constant parameter
        let dsize = data.len();
        let mut host_buffer =
            BufferView::<B>::new(device, memory_types, BufferType::HOST, dsize as u64, dtype);
        let mut device_buffer = BufferView::<B>::new(
            device,
            memory_types,
            BufferType::DEVICE,
            dsize as u64,
            dtype,
        );
        unsafe {
            // mapping => the handle at host side of device memory
            let mapping = device
                .map_memory(&mut host_buffer.memory, memory::Segment::ALL)
                .unwrap();
            ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                mapping,
                dsize * F32STRIDE as usize,
            );
            device.unmap_memory(&mut host_buffer.memory);
        }
        return Self {
            host_buffer: host_buffer,
            device_buffer: device_buffer,
            raw_data: data,
            data_size: dsize,
        };
    }

    pub fn eval(&mut self, device: &B::Device) {
        unsafe {
            let mapping = device
                .map_memory(&mut self.host_buffer.memory, memory::Segment::ALL)
                .unwrap();
            ptr::copy_nonoverlapping(
                mapping,
                self.raw_data.as_ptr() as *mut u8,
                self.data_size * F32STRIDE as usize,
            );
            device.unmap_memory(&mut self.host_buffer.memory);
        }
    }
}

// Buffer Functor that represent a collection of a buffer + memory object and its attributes
#[derive(Debug)]
pub(crate) struct BufferView<B: hal::Backend> {
    pub buffer_type: BufferType,
    pub buffer: B::Buffer,
    pub memory: B::Memory,
    pub buffer_size: u64,
    // TODO add shape
}

impl<'a, B: hal::Backend> BufferView<B> {
    pub fn new(
        device: &B::Device,
        memory_types: &[MemoryType],
        buffer_type: BufferType,
        data_size: u64,
        dtype: ElementType,
    ) -> Self {
        let stride = match dtype {
            ElementType::F32 => std::mem::size_of::<f32>() as buffer::Stride,
            ElementType::I32 => std::mem::size_of::<f64>() as buffer::Stride,
            _ => std::mem::size_of::<u8>() as buffer::Stride,
        };
        let properties = match buffer_type {
            BufferType::HOST => memory::Properties::CPU_VISIBLE | memory::Properties::COHERENT,
            BufferType::DEVICE => memory::Properties::DEVICE_LOCAL,
        };
        let usage = match buffer_type {
            BufferType::HOST => buffer::Usage::TRANSFER_SRC | buffer::Usage::TRANSFER_DST,
            BufferType::DEVICE => {
                buffer::Usage::TRANSFER_SRC | buffer::Usage::TRANSFER_DST | buffer::Usage::STORAGE
            }
        };
        let (memory, buffer, rsize) = unsafe {
            let mut buffer = device
                .create_buffer(
                    stride as u64 * data_size,
                    usage,
                    hal::memory::SparseFlags::empty(),
                )
                .unwrap();
            let requirements = device.get_buffer_requirements(&buffer);

            let ty = memory_types
                .into_iter()
                .enumerate()
                .position(|(id, memory_type)| {
                    requirements.type_mask & (1 << id) != 0
                        && memory_type.properties.contains(properties)
                })
                .unwrap()
                .into();

            let memory = device.allocate_memory(ty, requirements.size).unwrap();
            device.bind_buffer_memory(&memory, 0, &mut buffer).unwrap();
            (memory, buffer, requirements.size)
        };
        return Self {
            buffer_type: BufferType::DEVICE,
            buffer: buffer,
            memory: memory,
            buffer_size: rsize,
        };
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn create_buffer() {
        assert_eq!(0, 0);
    }
}
