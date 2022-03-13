// external crates
use nom::types::CompleteStr;
use nom::*;

// mods from local crate
use crate::instruction::OpCode;

use crate::assembler::assembler_base::Token;

named!(pub parse_operand <CompleteStr, Token>,
    ws!(
        do_parse!(
            tag!("%") >>
            //lab_symbol: alphanumeric1 >>
            lab_symbol: digit >>
            (
                Token::Variable{
                  symbol: lab_symbol.parse::<u8>().unwrap(),
                  // symbol: lab_symbol.parse::<String>().unwrap(),
                }
            )
        )
    )
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_operand() {
        let result = parse_operand(CompleteStr("%0"));
        assert_eq!(result.is_ok(), true);
        let _raw_result = result.unwrap().1;
        assert_eq!(_raw_result, Token::Variable { symbol: 0 as u8 });
        let result = parse_operand(CompleteStr("0"));
        assert_eq!(result.is_ok(), false);
        // TODO add label max id check
        let result = parse_operand(CompleteStr("%arg1"));
        assert_eq!(result.is_ok(), false);
    }
}
