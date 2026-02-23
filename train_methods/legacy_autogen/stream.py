from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol, Any, Iterator


class OutputStream(Protocol):
    def print(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        ...  # pragma: no cover


class InputStream(Protocol):
    def input(self, prompt: str = "", *, password: bool = False) -> str:
        ...  # pragma: no cover


class IOStream(InputStream, OutputStream, Protocol):

    # ContextVar must be used in multithreaded or async environments
    _default_io_stream: ContextVar["IOStream" | None] = ContextVar("default_iostream", default=None)
    _default_io_stream.set(None)
    _global_default: "IOStream" | None = None

    @staticmethod
    def set_global_default(stream: "IOStream") -> None:
        IOStream._global_default = stream

    @staticmethod
    def get_global_default() -> "IOStream":
        if IOStream._global_default is None:
            raise RuntimeError("No global default IOStream has been set")
        return IOStream._global_default

    @staticmethod
    def get_default() -> "IOStream":
        iostream = IOStream._default_io_stream.get()
        if iostream is None:
            iostream = IOStream.get_global_default()
            # Set the default IOStream of the current context (thread/cooroutine)
            IOStream.set_default(iostream)
        return iostream

    @staticmethod
    @contextmanager
    def set_default(stream: "IOStream" | None) -> Iterator[None]:
        global _default_io_stream
        try:
            token = IOStream._default_io_stream.set(stream)
            yield
        finally:
            IOStream._default_io_stream.reset(token)

        return
