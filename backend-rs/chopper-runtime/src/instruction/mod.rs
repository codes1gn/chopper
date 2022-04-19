use nom::types::CompleteStr;
use serde::{Deserialize, Serialize};

#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum OpCode {
    HALT, // 0
    LOAD, // 1

    ADDI32,      // 2
    SUBI32,      // 3
    MULI32,      // 4
    FLOORDIVI32, // 5

    // const literal command
    CONSTI32, // 6
    CONSTF32, // 7

    ADDF32,      // 8
    SUBF32,      // 9
    MULF32,      // 10
    DIVF32,      // 11
    CONSTTENSOR, // 12
    MATMULF32,   // 13

    // ILLEGAL op always id at last index
    ILLEGAL, // rest
}

impl OpCode {
    pub fn to_kernel_query_entry(&self) -> String {
        match self {
            // i32 types
            OpCode::ADDI32 | OpCode::SUBI32 | OpCode::MULI32 | OpCode::FLOORDIVI32 => {
                String::from("binary_arithmetic_i32")
            }

            // f32 types
            // TODO(tianyu), this file specify the kernel code file name
            OpCode::ADDF32 | OpCode::SUBF32 | OpCode::MULF32 | OpCode::DIVF32 => {
                String::from("binary_arithmetic_f32")
            }

            OpCode::MATMULF32 => String::from("matrix_multiple_f32"),

            _ => panic!("not support this op for dispatch kernel"),
        }
    }

    pub fn to_specialise_bits(&self) -> u32 {
        match self {
            // add spec data
            OpCode::ADDI32 | OpCode::ADDF32 => 0_u32,

            // sub spec data
            OpCode::SUBI32 | OpCode::SUBF32 => 1_u32,

            // sub spec data
            OpCode::MULI32 | OpCode::MULF32 => 2_u32,

            // floordiv
            OpCode::FLOORDIVI32 | OpCode::DIVF32 => 3_u32,

            // matrix-multiple
            OpCode::MATMULF32 => 4_u32,

            // TODO(tianyu): change matmul opcode into add opcode to fake the compute
            // OpCode::MATMULF32 => 0_u32,
            _ => panic!("unsupported opcode for specilising kernels"),
        }
    }
}

impl From<CompleteStr<'_>> for OpCode {
    fn from(s: CompleteStr<'_>) -> Self {
        match s {
            CompleteStr("halt") => OpCode::HALT,
            CompleteStr("load") => OpCode::LOAD,
            CompleteStr("crt.add.i32") => OpCode::ADDI32,
            CompleteStr("crt.sub.i32") => OpCode::SUBI32,
            CompleteStr("crt.mul.i32") => OpCode::MULI32,
            CompleteStr("crt.floordiv.i32") => OpCode::FLOORDIVI32,
            CompleteStr("crt.literal.const.i32") => OpCode::CONSTI32,
            CompleteStr("crt.literal.const.f32") => OpCode::CONSTF32,
            CompleteStr("crt.literal.const.tensor") => OpCode::CONSTTENSOR,
            CompleteStr("crt.add.f32") => OpCode::ADDF32,
            CompleteStr("crt.sub.f32") => OpCode::SUBF32,
            CompleteStr("crt.mul.f32") => OpCode::MULF32,
            CompleteStr("crt.matmul.f32") => OpCode::MATMULF32,
            CompleteStr("crt.div.f32") => OpCode::DIVF32,
            _ => OpCode::ILLEGAL,
        }
    }
}

// TODO
// ===================    arithmetic ops
// add
// sub
// mul
// div
// rem
// fma
// abs f32
// neg f32
// ceil f32
// floor f32
// atan f32
// atan2 f32
// cos f32
// sin f32
// exp f32
// exp2 f32
// expm1 f32
// log f32
// log10 f32
// log1p f32
// log2 f32
// pow f32
// rsprt f32
// sprt f32
// tanh f32
// NOT i32
// AND i32
// OR i32
// XOR i32
// ========================    casting ops
// bc_i32tof32
// bc_f32toi32
// =====================      shift ops
// shl
// shr
// ======================     invoke ops
// invoke
impl From<u8> for OpCode {
    fn from(v: u8) -> Self {
        match v {
            0 => {
                return OpCode::HALT;
            }
            1 => {
                return OpCode::LOAD;
            }
            2 => {
                return OpCode::ADDI32;
            }
            3 => {
                return OpCode::SUBI32;
            }
            4 => {
                return OpCode::MULI32;
            }
            5 => {
                return OpCode::FLOORDIVI32;
            }
            6 => {
                return OpCode::CONSTI32;
            }
            7 => {
                return OpCode::CONSTF32;
            }
            8 => {
                return OpCode::ADDF32;
            }
            9 => {
                return OpCode::SUBF32;
            }
            10 => {
                return OpCode::MULF32;
            }
            11 => {
                return OpCode::DIVF32;
            }
            12 => {
                return OpCode::CONSTTENSOR;
            }
            13 => {
                return OpCode::MATMULF32;
            }
            _ => {
                return OpCode::ILLEGAL;
            }
        }
    }
}

pub struct Instruction {
    opcode: OpCode,
}

// Note
impl Instruction {
    pub fn new(opcode: OpCode) -> Instruction {
        Instruction { opcode: opcode }
    }
}

#[cfg(test)]

mod tests {
    use super::*;

    #[test]
    fn test_opcodes() {
        let opcode = OpCode::HALT;
        assert_eq!(opcode, OpCode::HALT);

        let opcode = OpCode::LOAD;
        assert_eq!(opcode, OpCode::LOAD);

        let opcode = OpCode::ADDI32;
        assert_eq!(opcode, OpCode::ADDI32);

        let opcode = OpCode::SUBI32;
        assert_eq!(opcode, OpCode::SUBI32);

        let opcode = OpCode::MULI32;
        assert_eq!(opcode, OpCode::MULI32);

        let opcode = OpCode::FLOORDIVI32;
        assert_eq!(opcode, OpCode::FLOORDIVI32);

        let opcode = OpCode::CONSTI32;
        assert_eq!(opcode, OpCode::CONSTI32);

        let opcode = OpCode::CONSTF32;
        assert_eq!(opcode, OpCode::CONSTF32);

        let opcode = OpCode::CONSTTENSOR;
        assert_eq!(opcode, OpCode::CONSTTENSOR);
    }

    #[test]
    fn test_create_opcode() {
        let opcode = OpCode::ILLEGAL;
        let inst = Instruction::new(opcode);
        assert_eq!(inst.opcode, OpCode::ILLEGAL);

        let opcode = OpCode::CONSTF32;
        let inst = Instruction::new(opcode);
        assert_eq!(inst.opcode, OpCode::CONSTF32);

        let opcode = OpCode::CONSTI32;
        let inst = Instruction::new(opcode);
        assert_eq!(inst.opcode, OpCode::CONSTI32);

        let opcode = OpCode::CONSTTENSOR;
        let inst = Instruction::new(opcode);
        assert_eq!(inst.opcode, OpCode::CONSTTENSOR);

        let opcode = OpCode::MATMULF32;
        let inst = Instruction::new(opcode);
        assert_eq!(inst.opcode, OpCode::MATMULF32);
    }
}
