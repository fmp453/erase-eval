import base64
import json
import os
import re
import inspect
import importlib
import subprocess
import sys
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from hashlib import md5
from string import Template
from importlib.abc import SourceLoader
from queue import Empty
from textwrap import indent, dedent
from types import SimpleNamespace
from typing import Protocol, Literal, TypedDict, Mapping, Any, ClassVar, Callable, Generic, TypeVar
from typing_extensions import ParamSpec

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from pydantic import BaseModel, Field, field_validator


from pydantic import BaseModel, Field

from train_methods.legacy_autogen.utils import UserMessageImageContentPart, UserMessageTextContentPart, content_str, infer_lang, UNKNOWN, CODE_BLOCK_PATTERN, PYTHON_VARIANTS, WIN32, TIMEOUT_MSG, _cmd

A = ParamSpec("A")
T = TypeVar("T")
P = ParamSpec("P")

class CodeBlock(BaseModel):
    """(Experimental) A class that represents a code block."""

    code: str = Field(description="The code to execute.")

    language: str = Field(description="The language of the code.")

class CodeResult(BaseModel):
    """(Experimental) A class that represents the result of a code execution."""

    exit_code: int = Field(description="The exit code of the code execution.")

    output: str = Field(description="The output of the code execution.")


class CodeExtractor(Protocol):
    """(Experimental) A code extractor class that extracts code blocks from a message."""

    def extract_code_blocks(
        self, message: str | list[UserMessageTextContentPart | UserMessageImageContentPart] | None
    ) -> list[CodeBlock]:
        """(Experimental) Extract code blocks from a message.

        Args:
            message (str): The message to extract code blocks from.

        Returns:
            List[CodeBlock]: The extracted code blocks.
        """
        ...  # pragma: no cover

class CodeExecutor(Protocol):
    """(Experimental) A code executor class that executes code blocks and returns the result."""

    @property
    def code_extractor(self) -> CodeExtractor:
        """(Experimental) The code extractor used by this code executor."""
        ...  # pragma: no cover

    def execute_code_blocks(self, code_blocks: list[CodeBlock]) -> CodeResult:
        """(Experimental) Execute code blocks and return the result.

        This method should be implemented by the code executor.

        Args:
            code_blocks (List[CodeBlock]): The code blocks to execute.

        Returns:
            CodeResult: The result of the code execution.
        """
        ...  # pragma: no cover

    def restart(self) -> None:
        """(Experimental) Restart the code executor.

        This method should be implemented by the code executor.

        This method is called when the agent is reset.
        """
        ...  # pragma: no cover


class IPythonCodeResult(CodeResult):
    """(Experimental) A code result class for IPython code executor."""

    output_files: list[str] = Field(
        default_factory=list,
        description="The list of files that the executed code blocks generated.",
    )

class MarkdownCodeExtractor(CodeExtractor):
    """(Experimental) A class that extracts code blocks from a message using Markdown syntax."""

    def extract_code_blocks(
        self, message: str | list[UserMessageTextContentPart | UserMessageImageContentPart] | None
    ) -> list[CodeBlock]:
        """(Experimental) Extract code blocks from a message. If no code blocks are found,
        return an empty list.

        Args:
            message (str): The message to extract code blocks from.

        Returns:
            List[CodeBlock]: The extracted code blocks or an empty list.
        """

        text = content_str(message)
        match = re.findall(CODE_BLOCK_PATTERN, text, flags=re.DOTALL)
        if not match:
            return []
        code_blocks = []
        for lang, code in match:
            if lang == "":
                lang = infer_lang(code)
            if lang == UNKNOWN:
                lang = ""
            code_blocks.append(CodeBlock(code=code, language=lang))
        return code_blocks


CodeExecutionConfig = TypedDict(
    "CodeExecutionConfig",
    {
        "executor": Literal["ipython-embedded", "commandline-local"] | CodeExecutor,
        "last_n_messages": int | Literal["auto"],
        "timeout": int,
        "use_docker": bool | str | list[str],
        "work_dir": str,
        "ipython-embedded": Mapping[str, Any],
        "commandline-local": Mapping[str, Any],
    },
    total=False,
)


class EmbeddedIPythonCodeExecutor(BaseModel):
    """(Experimental) A code executor class that executes code statefully using an embedded
    IPython kernel managed by this class.

    **This will execute LLM generated code on the local machine.**

    Each execution is stateful and can access variables created from previous
    executions in the same session. The kernel must be installed before using
    this class. The kernel can be installed using the following command:
    `python -m ipykernel install --user --name {kernel_name}`
    where `kernel_name` is the name of the kernel to install.

    Args:
        timeout (int): The timeout for code execution, by default 60.
        kernel_name (str): The kernel name to use. Make sure it is installed.
            By default, it is "python3".
        output_dir (str): The directory to save output files, by default ".".
    """

    timeout: int = Field(default=60, ge=1, description="The timeout for code execution.")
    kernel_name: str = Field(default="python3", description="The kernel name to use. Make sure it is installed.")
    output_dir: str = Field(default=".", description="The directory to save output files.")

    @field_validator("output_dir")
    @classmethod
    def _output_dir_must_exist(cls, value: str) -> str:
        if not os.path.exists(value):
            raise ValueError(f"Output directory {value} does not exist.")
        return value

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # Check if the kernel is installed.
        if self.kernel_name not in KernelSpecManager().find_kernel_specs():
            raise ValueError(
                f"Kernel {self.kernel_name} is not installed. "
                "Please first install it with "
                f"`python -m ipykernel install --user --name {self.kernel_name}`."
            )
        self._kernel_manager = KernelManager(kernel_name=self.kernel_name)
        self._kernel_manager.start_kernel()
        self._kernel_client = self._kernel_manager.client()
        self._kernel_client.start_channels()
        self._timeout = self.timeout
        self._kernel_name = self.kernel_name
        self._output_dir = Path(self.output_dir)

    @property
    def code_extractor(self) -> CodeExtractor:
        """(Experimental) Export a code extractor that can be used by an agent."""
        return MarkdownCodeExtractor()

    def execute_code_blocks(self, code_blocks: list[CodeBlock]) -> IPythonCodeResult:
        """(Experimental) Execute a list of code blocks and return the result.

        This method executes a list of code blocks as cells in an IPython kernel
        managed by this class.
        See: https://jupyter-client.readthedocs.io/en/stable/messaging.html
        for the message protocol.

        Args:
            code_blocks (List[CodeBlock]): A list of code blocks to execute.

        Returns:
            IPythonCodeResult: The result of the code execution.
        """
        self._kernel_client.wait_for_ready()
        outputs = []
        output_files = []
        for code_block in code_blocks:
            code = self._process_code(code_block.code)
            self._kernel_client.execute(code, store_history=True)
            while True:
                try:
                    msg = self._kernel_client.get_iopub_msg(timeout=self._timeout)
                    msg_type = msg["msg_type"]
                    content = msg["content"]
                    if msg_type in ["execute_result", "display_data"]:
                        for data_type, data in content["data"].items():
                            if data_type == "text/plain":
                                # Output is a text.
                                outputs.append(data)
                            elif data_type.startswith("image/"):
                                # Output is an image.
                                path = self._save_image(data)
                                outputs.append(f"Image data saved to {path}")
                                output_files.append(path)
                            elif data_type == "text/html":
                                # Output is an html.
                                path = self._save_html(data)
                                outputs.append(f"HTML data saved to {path}")
                                output_files.append(path)
                            else:
                                # Output raw data.
                                outputs.append(json.dumps(data))
                    elif msg_type == "stream":
                        # Output is a text.
                        outputs.append(content["text"])
                    elif msg_type == "error":
                        # Output is an error.
                        return IPythonCodeResult(
                            exit_code=1,
                            output=f"ERROR: {content['ename']}: {content['evalue']}\n{content['traceback']}",
                        )
                    if msg_type == "status" and content["execution_state"] == "idle":
                        break
                # handle time outs.
                except Empty:
                    return IPythonCodeResult(
                        exit_code=1,
                        output=f"ERROR: Timeout waiting for output from code block: {code_block.code}",
                    )
        # We return the full output.
        return IPythonCodeResult(
            exit_code=0, output="\n".join([str(output) for output in outputs]), output_files=output_files
        )

    def restart(self) -> None:
        """(Experimental) Restart a new session."""
        self._kernel_client.stop_channels()
        self._kernel_manager.shutdown_kernel()
        self._kernel_manager = KernelManager(kernel_name=self.kernel_name)
        self._kernel_manager.start_kernel()
        self._kernel_client = self._kernel_manager.client()
        self._kernel_client.start_channels()

    def _save_image(self, image_data_base64: str) -> str:
        """Save image data to a file."""
        image_data = base64.b64decode(image_data_base64)
        # Randomly generate a filename.
        filename = f"{uuid.uuid4().hex}.png"
        path = os.path.join(self.output_dir, filename)
        with open(path, "wb") as f:
            f.write(image_data)
        return os.path.abspath(path)

    def _save_html(self, html_data: str) -> str:
        """Save html data to a file."""
        # Randomly generate a filename.
        filename = f"{uuid.uuid4().hex}.html"
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            f.write(html_data)
        return os.path.abspath(path)

    def _process_code(self, code: str) -> str:
        """Process code before execution."""
        # Find lines that start with `! pip install` and make sure "-qqq" flag is added.
        lines = code.split("\n")
        for i, line in enumerate(lines):
            # use regex to find lines that start with `! pip install` or `!pip install`.
            match = re.search(r"^! ?pip install", line)
            if match is not None:
                if "-qqq" not in line:
                    lines[i] = line.replace(match.group(0), match.group(0) + " -qqq")
        return "\n".join(lines)

class CommandLineCodeResult(CodeResult):
    """(Experimental) A code result class for command line code executor."""

    code_file: str | None = Field(
        default=None,
        description="The file that the executed code block was saved to.",
    )


@dataclass
class Alias:
    name: str
    alias: str


@dataclass
class ImportFromModule:
    module: str
    imports: list[str | Alias]

Import = str | ImportFromModule | Alias


class _StringLoader(SourceLoader):
    def __init__(self, data: str):
        self.data = data

    def get_source(self, fullname: str) -> str:
        return self.data

    def get_data(self, path: str) -> bytes:
        return self.data.encode("utf-8")

    def get_filename(self, fullname: str) -> str:
        return "<not a real path>/" + fullname + ".py"

@dataclass
class FunctionWithRequirementsStr:
    func: str
    _compiled_func: Callable[..., Any]
    _func_name: str
    python_packages: list[str] = field(default_factory=list)
    global_imports: list[Import] = field(default_factory=list)

    def __init__(self, func: str, python_packages: list[str] = [], global_imports: list[Import] = []):
        self.func = func
        self.python_packages = python_packages
        self.global_imports = global_imports

        module_name = "func_module"
        loader = _StringLoader(func)
        spec = importlib.util.spec_from_loader(module_name, loader)
        if spec is None:
            raise ValueError("Could not create spec")
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ValueError("Could not create loader")

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise ValueError(f"Could not compile function: {e}") from e

        functions = inspect.getmembers(module, inspect.isfunction)
        if len(functions) != 1:
            raise ValueError("The string must contain exactly one function")

        self._func_name, self._compiled_func = functions[0]

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("String based function with requirement objects are not directly callable")


@dataclass
class FunctionWithRequirements(Generic[T, P]):
    func: Callable[P, T]
    python_packages: list[str] = field(default_factory=list)
    global_imports: list[Import] = field(default_factory=list)

    @classmethod
    def from_callable(
        cls, func: Callable[P, T], python_packages: list[str] = [], global_imports: list[Import] = []
    ) -> "FunctionWithRequirements"[T, P]:
        return cls(python_packages=python_packages, global_imports=global_imports, func=func)

    @staticmethod
    def from_str(
        func: str, python_packages: list[str] = [], global_imports: list[Import] = []
    ) -> FunctionWithRequirementsStr:
        return FunctionWithRequirementsStr(func=func, python_packages=python_packages, global_imports=global_imports)

    # Type this based on F
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.func(*args, **kwargs)

def to_stub(func: Callable[..., Any] | FunctionWithRequirementsStr) -> str:
    """Generate a stub for a function as a string

    Args:
        func (Callable[..., Any]): The function to generate a stub for

    Returns:
        str: The stub for the function
    """
    if isinstance(func, FunctionWithRequirementsStr):
        return to_stub(func._compiled_func)

    content = f"def {func.__name__}{inspect.signature(func)}:\n"
    docstring = func.__doc__

    if docstring:
        docstring = dedent(docstring)
        docstring = '"""' + docstring + '"""'
        docstring = indent(docstring, "    ")
        content += docstring + "\n"

    content += "    ..."
    return content

def _to_code(func: FunctionWithRequirements[T, P] | Callable[P, T] | FunctionWithRequirementsStr) -> str:
    if isinstance(func, FunctionWithRequirementsStr):
        return func.func

    code = inspect.getsource(func)
    # Strip the decorator
    if code.startswith("@"):
        code = code[code.index("\n") + 1 :]
    return code

def _import_to_str(im: Import) -> str:
    if isinstance(im, str):
        return f"import {im}"
    elif isinstance(im, Alias):
        return f"import {im.name} as {im.alias}"
    else:

        def to_str(i: str | Alias) -> str:
            if isinstance(i, str):
                return i
            else:
                return f"{i.name} as {i.alias}"

        imports = ", ".join(map(to_str, im.imports))
        return f"from {im.module} import {imports}"

def _build_python_functions_file(
    funcs: list[FunctionWithRequirements[Any, P] | Callable[..., Any] | FunctionWithRequirementsStr]
) -> str:
    # First collect all global imports
    global_imports: set[str] = set()
    for func in funcs:
        if isinstance(func, (FunctionWithRequirements, FunctionWithRequirementsStr)):
            global_imports.update(map(_import_to_str, func.global_imports))

    content = "\n".join(global_imports) + "\n\n"

    for func in funcs:
        content += _to_code(func) + "\n\n"

    return content

filename_patterns = [
    re.compile(r"^<!-- (filename:)?(.+?) -->", re.DOTALL),
    re.compile(r"^/\* (filename:)?(.+?) \*/", re.DOTALL),
    re.compile(r"^// (filename:)?(.+?)$", re.DOTALL),
    re.compile(r"^# (filename:)?(.+?)$", re.DOTALL),
]

def _get_file_name_from_content(code: str, workspace_path: Path) -> str | None:
    first_line = code.split("\n")[0].strip()
    # TODO - support other languages
    for pattern in filename_patterns:
        matches = pattern.match(first_line)
        if matches is not None:
            filename = matches.group(2).strip()

            # Handle relative paths in the filename
            path = Path(filename)
            if not path.is_absolute():
                path = workspace_path / path
            path = path.resolve()
            # Throws an error if the file is not in the workspace
            relative = path.relative_to(workspace_path.resolve())
            return str(relative)
    return None

def silence_pip(code: str, lang: str) -> str:
    """Apply -qqq flag to pip install commands."""
    if lang == "python":
        regex = r"^! ?pip install"
    elif lang in ["bash", "shell", "sh", "pwsh", "powershell", "ps1"]:
        regex = r"^pip install"
    else:
        return code

    # Find lines that start with pip install and make sure "-qqq" flag is added.
    lines = code.split("\n")
    for i, line in enumerate(lines):
        # use regex to find lines that start with pip install.
        match = re.search(regex, line)
        if match is not None:
            if "-qqq" not in line:
                lines[i] = line.replace(match.group(0), match.group(0) + " -qqq")
    return "\n".join(lines)

class LocalCommandLineCodeExecutor(CodeExecutor):
    SUPPORTED_LANGUAGES: ClassVar[list[str]] = [
        "bash",
        "shell",
        "sh",
        "pwsh",
        "powershell",
        "ps1",
        "python",
        "javascript",
        "html",
        "css",
    ]
    DEFAULT_EXECUTION_POLICY: ClassVar[dict[str, bool]] = {
        "bash": True,
        "shell": True,
        "sh": True,
        "pwsh": True,
        "powershell": True,
        "ps1": True,
        "python": True,
        "javascript": False,
        "html": False,
        "css": False,
    }

    FUNCTION_PROMPT_TEMPLATE: ClassVar[
        str
    ] = """You have access to the following user defined functions. They can be accessed from the module called `$module_name` by their function names.

For example, if there was a function called `foo` you could import it by writing `from $module_name import foo`

$functions"""

    def __init__(
        self,
        timeout: int = 60,
        virtual_env_context: SimpleNamespace | None = None,
        work_dir: Path | str = Path("."),
        functions: list[FunctionWithRequirements[Any, A] | Callable[..., Any] | FunctionWithRequirementsStr] = [],
        functions_module: str = "functions",
        execution_policies: dict[str, bool] | None = None,
    ):
        """(Experimental) A code executor class that executes or saves LLM generated code a local command line
        environment.

        **This will execute or save LLM generated code on the local machine.**

        Each code block is saved as a file in the working directory. Depending on the execution policy,
        the code may be executed in a separate process.
        The code blocks are executed or save in the order they are received.
        Command line code is sanitized against a list of dangerous commands to prevent self-destructive commands from being executed,
        which could potentially affect the user's environment. Supported languages include Python, shell scripts (bash, shell, sh),
        PowerShell (pwsh, powershell, ps1), HTML, CSS, and JavaScript.
        Execution policies determine whether each language's code blocks are executed or saved only.

        ## Execution with a Python virtual environment
        A python virtual env can be used to execute code and install dependencies. This has the added benefit of not polluting the
        base environment with unwanted modules.
        ```python
        from autogen.code_utils import create_virtual_env
        from autogen.coding import LocalCommandLineCodeExecutor

        venv_dir = ".venv"
        venv_context = create_virtual_env(venv_dir)

        executor = LocalCommandLineCodeExecutor(virtual_env_context=venv_context)
        ```

        Args:
            timeout (int): The timeout for code execution, default is 60 seconds.
            virtual_env_context (Optional[SimpleNamespace]): The virtual environment context to use.
            work_dir (Union[Path, str]): The working directory for code execution, defaults to the current directory.
            functions (List[Union[FunctionWithRequirements[Any, A], Callable[..., Any], FunctionWithRequirementsStr]]): A list of callable functions available to the executor.
            functions_module (str): The module name under which functions are accessible.
            execution_policies (Optional[Dict[str, bool]]): A dictionary mapping languages to execution policies (True for execution, False for saving only). Defaults to class-wide DEFAULT_EXECUTION_POLICY.
        """

        if timeout < 1:
            raise ValueError("Timeout must be greater than or equal to 1.")

        if isinstance(work_dir, str):
            work_dir = Path(work_dir)

        if not functions_module.isidentifier():
            raise ValueError("Module name must be a valid Python identifier")

        self._functions_module = functions_module

        work_dir.mkdir(exist_ok=True)

        self._timeout = timeout
        self._work_dir: Path = work_dir
        self._virtual_env_context: SimpleNamespace | None = virtual_env_context

        self._functions = functions
        # Setup could take some time so we intentionally wait for the first code block to do it.
        if len(functions) > 0:
            self._setup_functions_complete = False
        else:
            self._setup_functions_complete = True

        self.execution_policies = self.DEFAULT_EXECUTION_POLICY.copy()
        if execution_policies is not None:
            self.execution_policies.update(execution_policies)

    def format_functions_for_prompt(self, prompt_template: str = FUNCTION_PROMPT_TEMPLATE) -> str:
        """(Experimental) Format the functions for a prompt.

        The template includes two variables:
        - `$module_name`: The module name.
        - `$functions`: The functions formatted as stubs with two newlines between each function.

        Args:
            prompt_template (str): The prompt template. Default is the class default.

        Returns:
            str: The formatted prompt.
        """
        template = Template(prompt_template)
        return template.substitute(
            module_name=self._functions_module,
            functions="\n\n".join([to_stub(func) for func in self._functions]),
        )

    @property
    def functions_module(self) -> str:
        """(Experimental) The module name for the functions."""
        return self._functions_module

    @property
    def functions(
        self,
    ) -> list[FunctionWithRequirements[Any, A] | Callable[..., Any] | FunctionWithRequirementsStr]:
        """(Experimental) The functions that are available to the code executor."""
        return self._functions

    @property
    def timeout(self) -> int:
        """(Experimental) The timeout for code execution."""
        return self._timeout

    @property
    def work_dir(self) -> Path:
        """(Experimental) The working directory for the code execution."""
        return self._work_dir

    @property
    def code_extractor(self) -> CodeExtractor:
        """(Experimental) Export a code extractor that can be used by an agent."""
        return MarkdownCodeExtractor()

    @staticmethod
    def sanitize_command(lang: str, code: str) -> None:
        """
        Sanitize the code block to prevent dangerous commands.
        This approach acknowledges that while Docker or similar
        containerization/sandboxing technologies provide a robust layer of security,
        not all users may have Docker installed or may choose not to use it.
        Therefore, having a baseline level of protection helps mitigate risks for users who,
        either out of choice or necessity, run code outside of a sandboxed environment.
        """
        dangerous_patterns = [
            (r"\brm\s+-rf\b", "Use of 'rm -rf' command is not allowed."),
            (r"\bmv\b.*?\s+/dev/null", "Moving files to /dev/null is not allowed."),
            (r"\bdd\b", "Use of 'dd' command is not allowed."),
            (r">\s*/dev/sd[a-z][1-9]?", "Overwriting disk blocks directly is not allowed."),
            (r":\(\)\{\s*:\|\:&\s*\};:", "Fork bombs are not allowed."),
        ]
        if lang in ["bash", "shell", "sh"]:
            for pattern, message in dangerous_patterns:
                if re.search(pattern, code):
                    raise ValueError(f"Potentially dangerous command detected: {message}")

    def _setup_functions(self) -> None:
        func_file_content = _build_python_functions_file(self._functions)
        func_file = self._work_dir / f"{self._functions_module}.py"
        func_file.write_text(func_file_content)

        # Collect requirements
        lists_of_packages = [x.python_packages for x in self._functions if isinstance(x, FunctionWithRequirements)]
        flattened_packages = [item for sublist in lists_of_packages for item in sublist]
        required_packages = list(set(flattened_packages))
        if len(required_packages) > 0:
            print("Ensuring packages are installed in executor.")
            if self._virtual_env_context:
                py_executable = self._virtual_env_context.env_exe
            else:
                py_executable = sys.executable
            cmd = [py_executable, "-m", "pip", "install"] + required_packages
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self._work_dir,
                    capture_output=True,
                    text=True,
                    timeout=float(self._timeout),
                    encoding="utf-8",
                )
            except subprocess.TimeoutExpired as e:
                raise ValueError("Pip install timed out") from e
            if result.returncode != 0:
                raise ValueError(f"Pip install failed. {result.stdout}, {result.stderr}")
        # Attempt to load the function file to check for syntax errors, imports etc.
        exec_result = self._execute_code_dont_check_setup([CodeBlock(code=func_file_content, language="python")])
        if exec_result.exit_code != 0:
            raise ValueError(f"Functions failed to load: {exec_result.output}")
        self._setup_functions_complete = True

    def execute_code_blocks(self, code_blocks: list[CodeBlock]) -> CommandLineCodeResult:
        """(Experimental) Execute the code blocks and return the result.

        Args:
            code_blocks (List[CodeBlock]): The code blocks to execute.

        Returns:
            CommandLineCodeResult: The result of the code execution."""
        if not self._setup_functions_complete:
            self._setup_functions()
        return self._execute_code_dont_check_setup(code_blocks)

    def _execute_code_dont_check_setup(self, code_blocks: list[CodeBlock]) -> CommandLineCodeResult:
        logs_all = ""
        file_names = []
        for code_block in code_blocks:
            lang, code = code_block.language, code_block.code
            lang = lang.lower()

            LocalCommandLineCodeExecutor.sanitize_command(lang, code)
            code = silence_pip(code, lang)

            if lang in PYTHON_VARIANTS:
                lang = "python"

            if WIN32 and lang in ["sh", "shell"]:
                lang = "ps1"

            if lang not in self.SUPPORTED_LANGUAGES:
                # In case the language is not supported, we return an error message.
                exitcode = 1
                logs_all += "\n" + f"unknown language {lang}"
                break

            execute_code = self.execution_policies.get(lang, False)
            try:
                # Check if there is a filename comment
                filename = _get_file_name_from_content(code, self._work_dir)
            except ValueError:
                return CommandLineCodeResult(exit_code=1, output="Filename is not in the workspace")

            if filename is None:
                # create a file with an automatically generated name
                code_hash = md5(code.encode()).hexdigest()
                filename = f"tmp_code_{code_hash}.{'py' if lang.startswith('python') else lang}"
            written_file = (self._work_dir / filename).resolve()
            with written_file.open("w", encoding="utf-8") as f:
                f.write(code)
            file_names.append(written_file)

            if not execute_code:
                # Just return a message that the file is saved.
                logs_all += f"Code saved to {str(written_file)}\n"
                exitcode = 0
                continue

            program = _cmd(lang)
            cmd = [program, str(written_file.absolute())]
            env = os.environ.copy()

            if self._virtual_env_context:
                virtual_env_abs_path = os.path.abspath(self._virtual_env_context.bin_path)
                path_with_virtualenv = rf"{virtual_env_abs_path}{os.pathsep}{env['PATH']}"
                env["PATH"] = path_with_virtualenv
                if WIN32:
                    activation_script = os.path.join(virtual_env_abs_path, "activate.bat")
                    cmd = [activation_script, "&&", *cmd]

            try:
                result = subprocess.run(
                    cmd,
                    cwd=self._work_dir,
                    capture_output=True,
                    text=True,
                    timeout=float(self._timeout),
                    env=env,
                    encoding="utf-8",
                )
            except subprocess.TimeoutExpired:
                logs_all += "\n" + TIMEOUT_MSG
                # Same exit code as the timeout command on linux.
                exitcode = 124
                break

            logs_all += result.stderr
            logs_all += result.stdout
            exitcode = result.returncode

            if exitcode != 0:
                break

        code_file = str(file_names[0]) if len(file_names) > 0 else None
        return CommandLineCodeResult(exit_code=exitcode, output=logs_all, code_file=code_file)

    def restart(self) -> None:
        """(Experimental) Restart the code executor."""
        warnings.warn("Restarting local command line code executor is not supported. No action is taken.")
