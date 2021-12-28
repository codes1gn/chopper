"""
Simple JSON Parser
==================
The code is short and clear, and outperforms every other parser (that's written in Python).
For an explanation, check out the JSON parser tutorial at /docs/json_tutorial.md
(this is here for use by the other examples)
"""
import sys

from lark import Lark, Transformer, v_args

__all__ = [
    'json_grammer',
    'TreeToJson',
]

json_grammar = r"""
    ?start: value
    ?value: object
          | array
          | string
          | SIGNED_NUMBER      -> number
          | "true"             -> true
          | "false"            -> false
          | "null"             -> null
    array  : "[" [value ("," value)*] "]"
    object : "{" [pair ("," pair)*] "}"
    pair   : string ":" value
    string : ESCAPED_STRING
    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
"""


class TreeToJson(Transformer):
    @v_args(inline=True)
    def string(self, s):
        return s[1:-1].replace('\\"', '"')

    array = list
    pair = tuple
    object = dict
    number = v_args(inline=True)(float)

    def null(self, _):
        return None

    def true(self, _):
        return True

    def false(self, _):
        return False


# Create the JSON parser with Lark, using the LALR algorithm
json_parser = Lark(
    json_grammar,
    parser='lalr',
    # Using the standard lexer isn't required, and isn't usually recommended.
    # But, it's good enough for JSON, and it's slightly faster.
    lexer='standard',
    # Disabling propagate_positions and placeholders slightly
    # improves speed
    propagate_positions=False,
    maybe_placeholders=False,
    # Using an internal transformer is faster and more memory
    # efficient
    transformer=TreeToJson())
