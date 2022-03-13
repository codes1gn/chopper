// external crates
use nom::types::CompleteStr;
use nom::*;

// mods from local crate
use crate::instruction::OpCode;

use super::assembler_base::*;
use super::parse_instruction::*;

named!(pub parse_program<CompleteStr, Program>,
    do_parse!(
        instructions: many1!(
            parse_instruction
        ) >> (
            Program {
                instructions: instructions
            }
        )
    )
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_program_const_literal() {
        // w. \n
        let result = parse_program(CompleteStr("%0 = crt.literal.const.i32! 13 : i32\n"));
        println!("{:?}", result);
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1.to_bytes();
        assert_eq!(_bytes_result, vec![6, 0, 13, 0, 0, 0])
    }
}
