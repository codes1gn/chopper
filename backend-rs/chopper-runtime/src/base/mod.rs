pub mod constants;
pub mod errors;
pub mod kernel;

use nom::types::CompleteStr;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ElementType {
    I32,
    F32,
}

impl From<CompleteStr<'_>> for ElementType {
    fn from(s: CompleteStr<'_>) -> Self {
        match s {
            CompleteStr("i32") => ElementType::I32,
            CompleteStr("f32") => ElementType::F32,
            _ => panic!("not recognise this element type"),
        }
    }
}

pub trait SupportedType {
    fn get_type_code(&self) -> ElementType;
}

impl SupportedType for i32 {
    fn get_type_code(&self) -> ElementType {
        return ElementType::I32;
    }
}

impl SupportedType for f32 {
    fn get_type_code(&self) -> ElementType {
        return ElementType::F32;
    }
}
