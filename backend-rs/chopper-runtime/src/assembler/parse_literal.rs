// external crates
use nom::types::CompleteStr;
use nom::*;

use super::assembler_base::*;
use super::parse_type::*;

// numeric-literal ::= integer-literal | float-literal
named!(pub parse_numeric_literal<CompleteStr, Token>,
    do_parse!(
        token: alt!(
            parse_float_literal | parse_integer_literal
            // TODO, match float first then integer, parse_integer_literal | parse_float_literal
        ) >>
        ( token )
    )
);

// "dense<[1.1 2.2 3.3 4.4 5.5 6.6], shape=[2 3]>"
named!(pub parse_tensor_literal<CompleteStr, Token>,
    do_parse!(
        _s: space0 >>
        tag!("dense<") >>
        datalist: parse_float_list >>
        tag!(", shape=") >>
        shapelist: parse_integer_list >>
        tag!(">") >>
        (
            Token::Tensor { raw_data: datalist, shape: shapelist }
        )
    )
);

named!(pub parse_float_list<CompleteStr, Vec<f32>>,
    do_parse!(
        _s: space0 >>
        tag!("[") >>
        data: many1!(
            parse_raw_float_literal
        ) >>
        tag!("]") >>
        (
            data
        )
    )
);

named!(pub parse_raw_float_literal<CompleteStr, f32>,
    do_parse!(
        _s: space0 >>
        data: float >>
        (
            data
        )
    )
);

named!(pub parse_integer_list<CompleteStr, Vec<i32>>,
    do_parse!(
        _s: space0 >>
        tag!("[") >>
        data: many1!(
            parse_raw_integer_literal
        ) >>
        tag!("]") >>
        (
            data
        )
    )
);

named!(pub parse_raw_integer_literal<CompleteStr, i32>,
    do_parse!(
        _s: space0 >>
        data: digit >>
        (
            data.parse::<i32>().unwrap()
        )
    )
);

named!(pub parse_integer_literal<CompleteStr, Token>,
    do_parse!(
        _s: space0 >>
        data: digit >>
        (
            Token::I32Literal { value: data.parse::<i32>().unwrap() }
        )
    )
);

named!(pub parse_integer_literal_with_type<CompleteStr, Token>,
    do_parse!(
        _s: space0 >>
        data: digit >>
        type_tag: parse_i32_type >>
        (
            Token::I32Literal { value: data.parse::<i32>().unwrap() }
        )
    )
);

named!(pub parse_float_literal<CompleteStr, Token>,
    do_parse!(
        _s: space0 >>
        data: float >>
        (
            Token::F32Literal { value: data }
        )
    )
);

named!(pub parse_float_literal_with_type<CompleteStr, Token>,
    do_parse!(
        _s: space0 >>
        data: float >>
        type_tag: parse_f32_type >>
        (
            Token::F32Literal { value: data }
        )
    )
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_integer_literal() {
        // w.o. \n
        let result = parse_integer_literal(CompleteStr("23"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(_bytes_result, Token::I32Literal { value: 23 });

        // w. \n
        let result = parse_integer_literal(CompleteStr(" 23\n"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(_bytes_result, Token::I32Literal { value: 23 });
    }

    #[test]
    fn test_parse_raw_integer_literal() {
        // w.o. \n
        let result = parse_raw_integer_literal(CompleteStr("23"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(_bytes_result, 23);

        // w. \n
        let result = parse_raw_integer_literal(CompleteStr(" 23\n"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(_bytes_result, 23);
    }

    #[test]
    fn test_parse_integer_list() {
        // w.o. \n
        let result = parse_integer_list(CompleteStr("[23 2]"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(_bytes_result, vec![23, 2]);

        // w. \n
        let result = parse_integer_list(CompleteStr(" [23 2]\n"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(_bytes_result, vec![23, 2]);
    }

    #[test]
    fn test_parse_float_literal() {
        // w.o. \n
        let result = parse_float_literal(CompleteStr("23.7"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(_bytes_result, Token::F32Literal { value: 23.7 });

        // w. \n
        let result = parse_float_literal(CompleteStr("23.4\n"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(_bytes_result, Token::F32Literal { value: 23.4 });
    }

    #[test]
    fn test_parse_raw_float_literal() {
        // w.o. \n
        let result = parse_raw_float_literal(CompleteStr("2.3"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(_bytes_result, 2.3);

        // w. \n
        let result = parse_raw_float_literal(CompleteStr(" 2.3\n"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(_bytes_result, 2.3);
    }

    #[test]
    fn test_parse_float_list() {
        // w.o. \n
        let result = parse_float_list(CompleteStr("[2.3 2.1]"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(_bytes_result, vec![2.3, 2.1]);

        // w. \n
        let result = parse_float_list(CompleteStr(" [2.3 2.2]\n"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(_bytes_result, vec![2.3, 2.2]);
    }

    #[test]
    fn test_parse_tensor_literal() {
        // w.o. \n
        let result = parse_tensor_literal(CompleteStr("dense<[1.1 2.2 3.3 4.4 5.5 6.6], shape=[2 3]>"));
        assert_eq!(result.is_ok(), true);
        let _bytes_result = result.unwrap().1;
        assert_eq!(_bytes_result, Token::Tensor { raw_data: vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6], shape: vec![2, 3] });
    }
}
