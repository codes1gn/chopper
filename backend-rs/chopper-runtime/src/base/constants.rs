extern crate hal;

use hal::buffer;

// const attributes for float data types
pub const F32STRIDE: buffer::Stride = std::mem::size_of::<f32>() as buffer::Stride;

// const attributes for signed integer data types
pub const I32STRIDE: buffer::Stride = std::mem::size_of::<i32>() as buffer::Stride;

// const attributes for unsigned integer data types
pub const U8STRIDE: buffer::Stride = std::mem::size_of::<u8>() as buffer::Stride;

