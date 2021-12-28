import os
import ast
import astunparse

__all__ = ["ColorPalette"]


class ColorPalette:
    HEADER = "\033[95m"
    WARNING = "\033[93m"
    ENDC = "\033[0m"
    FAIL = "\033[91m"
    HEADER = "\033[95m"
