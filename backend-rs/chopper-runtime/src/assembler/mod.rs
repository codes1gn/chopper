// external crates
use nom::types::CompleteStr;
use nom::*;

// mods from local crate
use crate::instruction::OpCode;

// submods
pub mod assembler_base;
pub mod parse_instruction;
pub mod parse_literal;
pub mod parse_module;
pub mod parse_opcode;
pub mod parse_operand;
pub mod parse_type;

use assembler_base::*;
use parse_module::*;

named!(pub parse_bytecode<CompleteStr, Program>,
    do_parse!(
        module: parse_program >> (
            module
        )
    )
);

#[cfg(test)]
mod tests {
    use super::*;
}
