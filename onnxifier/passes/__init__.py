"""
Copyright (C) 2026 The ONNXIFIER Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import csv
import inspect
import io
import json
import shutil
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import chain
from pathlib import Path
from typing import Optional, Protocol, TypeVar, cast

from tabulate import tabulate

from ..traits import RewriterInterface
from .auto_load import auto_load
from .rewriter import Rewriter


class GraphNode(Protocol):
    """Any node to be registered in the Registry shall follow this protocol."""

    __DEPS__: list[str | type[Rewriter]]
    __PATCHES__: list[str | type[Rewriter]]


T = TypeVar("T", bound=GraphNode)
F = TypeVar("F", bound=Callable)


def _longest_line(cells: Iterable[object]) -> int:
    """Width a column needs to show its content unwrapped."""
    return max(
        (len(line) for cell in cells for line in str(cell).splitlines()), default=0
    )


class FuncInterfaceWrapper[T: GraphNode]:
    def __init__(
        self,
        func: Callable,
        name: str | None,
        deps: list[str | type[Rewriter]] | None,
        patches: list[str | type[Rewriter]] | None,
    ):
        # pylint: disable=invalid-name
        self.__FUNC = func
        self.__NAME__ = name or func.__name__
        self.__DEPS__ = deps or []
        self.__PATCHES__ = patches or []
        self.__DOC__ = inspect.getdoc(func)
        setattr(func, "__NAME__", self.__NAME__)
        setattr(func, "__DEPS__", self.__DEPS__)
        setattr(func, "__PATCHES__", self.__PATCHES__)
        setattr(func, "__DOC__", self.__DOC__)

    def __call__(self) -> T:
        return cast(T, self.__FUNC)


class Registry[T: GraphNode]:
    """A simple registry object to hold objects from others

    Samples::

        FOO = Registry("FOO")

        @FOO.register()
        def foo(): ...

        print(FOO)
        # ┌───────────────┐
        # │ Register: FOO │
        # ├───────────────┤
        # │ foo           │
        # └───────────────┘
    """

    def __init__(self, name=None, parent: Optional["Registry[T]"] = None) -> None:
        self._bucks: dict[str, type[T] | FuncInterfaceWrapper[T]] = {}
        self._configs: dict = {}
        self._docs: dict[str, str | None] = {}
        self._name = name or "<Registry>"
        self._parent = parent
        if parent is not None:
            self._name = f"{parent.name}.{self.name}"

    @property
    def name(self) -> str:
        """Return the name of the registry."""
        return self._name

    @staticmethod
    def _legal_name(name: str) -> str:
        words = [""]
        for a, b in zip(list(name), list(name.lower())):
            if a != b:
                words.append("")
            words[-1] += b
        return "_".join(words).strip("_")

    def register(
        self,
        name: str | None = None,
        deps: list[str | type[Rewriter]] | None = None,
        patch: list[str | type[Rewriter]] | None = None,
    ):
        """A decorator to register an object.

        Args:
            name (str, optional): The name of the object. If not provided, the name
                of the function of class will be used after transform to lowercase.
            deps (List[str], optional): The dependencies before executing the object.
            patch (List[str], optional): The hook after the object execution.
        """

        def wrapper(func: F) -> F:
            if not callable(func):
                raise TypeError(
                    "the object to be registered must be a function or Rewriter,"
                    f" got {type(func)}"
                )
            if inspect.isfunction(func):
                func_wrap = FuncInterfaceWrapper[T](func, name, deps, patch)
                self._bucks[func_wrap.__NAME__] = func_wrap
                self._configs[func_wrap.__NAME__] = inspect.signature(func)
                self._docs[func_wrap.__NAME__] = func_wrap.__DOC__
            else:
                assert isinstance(func, type)
                if not issubclass(func, Rewriter):
                    raise TypeError(
                        f"the registered object {func} must be the subclass "
                        f"of Rewriter, but its mro is {func.__mro__}"
                    )

                # note name is not saved because obj is gc-ed after this function
                func.__NAME__ = name or self._legal_name(func.__name__)
                func.__DEPS__.extend(deps or [])
                func.__PATCHES__.extend(patch or [])
                self._bucks[func.__NAME__] = cast(type[T], func)
                self._configs[func.__NAME__] = inspect.signature(func.rewrite)
                self._docs[func.__NAME__] = inspect.getdoc(func)
            if self._parent is not None:
                self._parent.register(name, deps, patch)(func)
            # forward the signature of the original function
            return cast(F, func)

        return wrapper

    def get(self, name: str | type[T]) -> T | None:
        """Get a registered object by its name."""
        if inspect.isclass(name):
            return name()
        if name in self._bucks:
            functor = self._bucks[name]()  # create a new instance each time
            # functor.__NAME__ = name  # rename the instance
            return functor

    def get_type(self, name: str | type[T]):
        """Get a registered object type by its name."""
        if inspect.isclass(name):
            return name
        if name in self._bucks:
            functor = self._bucks[name]  # create a new instance each time
            # functor.__NAME__ = name  # rename the instance
            return functor

    def get_config(self, name: str):
        """Get the configuration of an object"""
        return self._configs.get(name)

    def child(self, passes: str | Sequence[str]) -> "Registry":
        """Slice a child registry by given a set of pass names."""

        reg = self.__class__(parent=self)
        if isinstance(passes, str):
            passes = [passes]
        # pylint: disable=protected-access
        reg._bucks = {k: self._bucks[k] for k in passes}
        reg._configs = {k: self._configs[k] for k in passes}
        reg._docs = {k: self._docs[k] for k in passes}
        return reg

    def __getitem__(self, name: str | type[T]) -> T:
        """Get a registered object by its name."""
        obj = self.get(name)
        if obj is None:
            raise KeyError(f"{name} is not registered in {self._name}")
        return obj

    def __iter__(self) -> Iterator[str]:
        """Return an Iterator for all registered functions"""
        yield from self._bucks.keys()

    def __contains__(self, name: str) -> bool:
        """Check if a function is registered"""
        return name in self._bucks

    def __repr__(self) -> str:
        title = [f"Register: {self._name}", "Deps", "Patch", "Config"]
        members = []
        for k in sorted(self._bucks.keys()):
            members.append(
                [
                    k,
                    self._bucks[k].__DEPS__,
                    self._bucks[k].__PATCHES__,
                    self._configs[k],
                ]
            )
        return tabulate(members, title, "simple_grid", maxcolwidths=[None, 50, 50, 50])

    def to_format(self, fmt: str = "table", full: bool = False) -> str:
        """Render the registry as a string in the requested format.

        Args:
            fmt (str): ``table`` (default), ``csv`` or ``json``.
            full (bool): include the rewriter docstring (manual) column.
        """

        headers = ["PASS", "DEPS", "PATCH", "CONFIG"]
        if full:
            headers.append("DOC")
        members: list[list] = []
        for k in sorted(self._bucks.keys()):
            deps = [
                i.__NAME__ if inspect.isclass(i) else i for i in self._bucks[k].__DEPS__
            ]
            patches = [
                i.__NAME__ if inspect.isclass(i) else i
                for i in self._bucks[k].__PATCHES__
            ]
            row: list = [k, deps, patches, str(self._configs[k])]
            if full:
                row.append(self._docs.get(k) or "")
            members.append(row)
        if fmt == "json":
            records = [dict(zip(headers, row)) for row in members]
            return json.dumps(records, indent=2, ensure_ascii=False)
        if fmt == "csv":
            # tabulate 0.10 has no "csv" fmt; write it manually so fields are
            # properly quoted (DEPS/PATCH contain commas when joined).
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(headers)
            for row in members:
                writer.writerow(
                    [
                        ",".join(cells) if isinstance(cells, list) else cells
                        for cells in row
                    ]
                )
            return buf.getvalue().rstrip("\n")
        # table: content-aware widths, adaptive to the terminal (COLUMNS env
        # overrides). The PASS column is never wrapped; the remaining columns
        # take their natural width if it fits, else the excess is shed evenly
        # from the widest ones (water-filling).
        term = shutil.get_terminal_size((120, 24)).columns
        overhead = 3 * len(headers) + 1
        pass_w = _longest_line(chain([headers[0]], (m[0] for m in members)))
        widths = [
            _longest_line(chain([headers[i]], (m[i] for m in members)))
            for i in range(1, len(headers))
        ]
        budget = max(30, term - overhead - pass_w)
        excess = sum(widths) - budget
        while excess > 0 and any(w > 10 for w in widths):
            widths[widths.index(max(widths))] -= 1
            excess -= 1
        maxcolwidths: list[int | None] = [None, *widths]
        return tabulate(members, headers, "simple_grid", maxcolwidths=maxcolwidths)


def get_pass_manager(
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    configs: dict[str, dict[str, str | int | float | bool]] | None = None,
):
    """Lazy load pass manager"""
    # pylint: disable=import-outside-toplevel
    from ..pass_manager import PassManager

    return PassManager(include, exclude, configs)


PASSES = Registry[RewriterInterface]("PASS")
L1 = Registry[RewriterInterface]("L1", parent=PASSES)
L2 = Registry[RewriterInterface]("L2", parent=PASSES)
L3 = Registry[RewriterInterface]("L3", parent=PASSES)

_AUTO_LOAD_FOLDERS = sorted(filter(Path.is_dir, Path(__file__).parent.glob("*")))
for i in _AUTO_LOAD_FOLDERS:
    auto_load(i)
