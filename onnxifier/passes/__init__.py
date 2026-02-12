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

import inspect
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Type,
    TypeVar,
    cast,
)

from tabulate import tabulate

from ..traits import RewriterInterface
from .auto_load import auto_load
from .rewriter import Rewriter


class GraphNode(Protocol):
    """Any node to be registered in the Registry shall follow this protocol."""

    __DEPS__: List[str | Type[Rewriter]]
    __PATCHES__: List[str | Type[Rewriter]]


T = TypeVar("T", bound=GraphNode)
F = TypeVar("F", bound=Callable)


class FuncInterfaceWrapper(Generic[T]):

    def __init__(
        self,
        func: Callable,
        name: Optional[str],
        deps: Optional[List[str | Type[Rewriter]]],
        patches: Optional[List[str | Type[Rewriter]]],
    ):
        # pylint: disable=invalid-name
        self.__FUNC = func
        self.__NAME__ = name or func.__name__
        self.__DEPS__ = deps or []
        self.__PATCHES__ = patches or []
        setattr(func, "__NAME__", self.__NAME__)
        setattr(func, "__DEPS__", self.__DEPS__)
        setattr(func, "__PATCHES__", self.__PATCHES__)

    def __call__(self) -> T:
        return cast(T, self.__FUNC)


class Registry(Generic[T]):
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
        self._bucks: Dict[str, Type[T] | FuncInterfaceWrapper[T]] = {}
        self._configs: Dict = {}
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
        name: Optional[str] = None,
        deps: Optional[List[str | Type[Rewriter]]] = None,
        patch: Optional[List[str | Type[Rewriter]]] = None,
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
                self._bucks[func.__NAME__] = cast(Type[T], func)
                self._configs[func.__NAME__] = inspect.signature(func.rewrite)
            if self._parent is not None:
                self._parent.register(name, deps, patch)(func)
            # forward the signature of the original function
            return cast(F, func)

        return wrapper

    def get(self, name: str | Type[T]) -> Optional[T]:
        """Get a registered object by its name."""
        if inspect.isclass(name):
            return name()
        if name in self._bucks:
            functor = self._bucks[name]()  # create a new instance each time
            # functor.__NAME__ = name  # rename the instance
            return functor

    def get_type(self, name: str | Type[T]):
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
        return reg

    def __getitem__(self, name: str | Type[T]) -> T:
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


def get_pass_manager(
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
    configs: Optional[Dict[str, Dict[str, str | int | float | bool]]] = None,
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
