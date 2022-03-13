// external crates
use nom::types::CompleteStr;
use nom::*;

use crate::assembler::assembler_base::*;
use crate::base::*;

named!(pub parse_type<CompleteStr, Token>,
    do_parse!(
        _s: space0 >>
        token: alt!(
            tag!("i32") | tag!("f32")
        ) >>
        ( Token::DType { element_type: ElementType::from(token) } )
    )
);

// TODO hardcode for temp, make it into combined type indicator and type annotation.
named!(pub parse_i32_type<CompleteStr, Token>,
    do_parse!(
        _s: space0 >>
        tag!(": i32") >>
        ( Token::DType { element_type: ElementType::I32 } )
    )
);

named!(pub parse_f32_type<CompleteStr, Token>,
    do_parse!(
        _s: space0 >>
        tag!(": f32") >>
        ( Token::DType { element_type: ElementType::F32 } )
    )
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_type() {
        // w.o. \n
        let result = parse_type(CompleteStr("i32"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(
            _bytes_result,
            Token::DType {
                element_type: ElementType::I32
            }
        );

        // w. \n
        let result = parse_type(CompleteStr(" f32\n"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(
            _bytes_result,
            Token::DType {
                element_type: ElementType::F32
            }
        );
    }
}
