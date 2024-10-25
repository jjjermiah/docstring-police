import ast as _ast
from typing import List, Optional
from pathlib import Path
from unittest import result


class CodeBlockExtractor:
    """
    A class to extract functions and class methods from a Python file.

    This class reads a .py file and extracts each of the functions and class methods from the file.
    It returns a list of strings, where each string contains the entire code for a function or method,
    including the definition, docstring, and code. If a specific code block name is provided,
    only that block will be extracted.

    Attributes
    ----------
    file_path : Path
        The path to the .py file to extract functions and class methods from.
    content : str
        The content of the Python file read from the given file path.

    Methods
    -------
    extract_code_blocks(code_block_name: str = "") -> List[Optional[str]]:
        Extracts all functions and class methods, or a specific code block if a name is provided.
    """

    __slots__ = "_file_path", "_content", "_ast_result", "_code_blocks"

    def __init__(self, file_path: Path) -> None:
        """
        Initializes the CodeBlockExtractor with the specified file path.

        Parameters
        ----------
        file_path : Path
            The path to the .py file to extract functions and class methods from.
        """
        self._file_path: Path = file_path
        self._content = self._read_file_content()
        self._ast_result = _ast.parse(self._content)
        self._code_blocks = self._extract_functions_and_methods()

    @property
    def ast(self) -> _ast.Module:
        """The AST of the Python file."""
        return self._ast_result

    @property
    def content(self) -> str:
        """The content of the Python file."""
        return self._content

    @property
    def file_path(self) -> Path:
        """The path to the Python file."""
        return self._file_path

    @property
    def code_blocks(self) -> List[_ast.AST]:
        """The list of code blocks extracted from the Python file."""
        return self._code_blocks

    def _read_file_content(self) -> str:
        """Helper method to read file content from the file path."""
        try:
            return self.file_path.read_text()
        except FileNotFoundError:
            raise FileNotFoundError(f"The file at {self.file_path} was not found.")
        except PermissionError:
            raise PermissionError(
                f"Permission denied when trying to read {self.file_path}."
            )
        except Exception as e:
            raise RuntimeError(f"An error occurred while reading the file: {e}")

    @property
    def functions(self) -> List[_ast.FunctionDef]:
        """The list of functions extracted from the Python file."""
        return [
            node for node in self._code_blocks if isinstance(node, _ast.FunctionDef)
        ]

    @property
    def classes(self) -> List[_ast.ClassDef]:
        """The list of classes extracted from the Python file."""
        return [
            node for node in self._code_blocks if isinstance(node, _ast.ClassDef)
        ]

    def _extract_functions_and_methods(
        self, code_block_name: Optional[str] = None
    ) -> List[_ast.AST]:
        """Helper method to extract functions and methods from the AST."""
        code_blocks = []

        for node in self._ast_result.body:
            if isinstance(node, _ast.FunctionDef) and (
                code_block_name is None or node.name == code_block_name
            ):
                code_blocks.append(node)
            elif isinstance(node, _ast.ClassDef):
                # Add the class itself if requested by name
                code_blocks.append(node)
            else:
                # print(f"Skipping node of type {type(node)}")
                continue

        return code_blocks

    @staticmethod
    def get_docstring(node: _ast.AST) -> str | None:
        """Helper method to extract the docstring from a given node."""
        if not isinstance(node, (_ast.AsyncFunctionDef, _ast.FunctionDef, _ast.ClassDef)):
            raise TypeError(f"Expected a function or class definition, got {type(node)}")
        
        return _ast.get_docstring(node, clean=True)


if __name__ == "__main__":
    from rich import print

    example_dir = Path(__file__).parent.parent.parent / "tests" / "examples"
    for i, f in enumerate(example_dir.glob("*.py")):
        print(f"Extracting code blocks from {f.name}")
        result = CodeBlockExtractor(f).extract_code_blocks()

        print(result)
        if i == 2:
            break

# TODO: extract better
# assignees: jjjermiah