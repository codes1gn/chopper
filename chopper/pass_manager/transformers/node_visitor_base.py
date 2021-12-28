import ast
from mlir import astnodes

MlirNode = astnodes.Node


class NodeVisitorBase(ast.NodeVisitor):
    """This is a Class that inherit ast.NodeVisitor to pretty dump mlir ast node.

    Attributes:
        None.
    """
    def __init__(self):
        """initialize the NodevisitorBase class
        """
        super().__init__()

    def pretty_mlir(self, node: MlirNode) -> str:
        """pretty dump mlir ast node that convenient to check

        Args:
            node (MlirNode): The mlir ast node.

        Returns:
            str: pretty dumped ast node string.
        """
        result = node.dump_ast()
        lines = [""]
        indent = 0
        for index in range(len(result)):
            char = result[index]
            indent_word = "  "

            if char == " ":
                continue

            if char == "[" and result[index + 1] == "]":
                indent += 1
                lines[-1] += char
                continue

            if char == ",":
                lines[-1] += char
                lines.append(indent * indent_word)
                continue

            if char == "[":
                indent += 1
                lines[-1] += char
                lines.append(indent * indent_word)
                continue
            if char == "]":
                indent -= 1

            if char == "(":
                indent += 1
                lines[-1] += char
                lines.append(indent * "  ")
                continue
            if char == ")":
                indent -= 1

            if char != "\n":
                lines[-1] += char
            if char == "\n":
                lines.append(indent * indent_word)

        return "\n".join(lines)
