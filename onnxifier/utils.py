"""
Copyright (C) 2025 The ONNXIFIER Authors.

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

from contextlib import contextmanager
from os import PathLike
from os import chdir as _chdir
from pathlib import Path
from typing import Optional


def legalize_name(name: str) -> str:
    """Replace illegal characters in name with underscores."""
    name = (
        f"{name}".replace(":", "_")
        .replace("/", "_")
        .replace("?", "_")
        .replace("*", "_")
    )
    return name


def legalize_path_name(name: str | PathLike | Path) -> Path:
    """Replace illegal characters in path name with underscores."""
    parent = Path(name).parent
    name = Path(name).name
    name = legalize_name(name)
    return parent / Path(name)


@contextmanager
def chdir(path: Optional[str | PathLike | Path]):
    """Context manager to change current working directory."""
    if path is None:
        yield
        return
    old_cwd = Path.cwd()
    try:
        _chdir(path)
        yield Path(path).resolve()
    finally:
        _chdir(old_cwd)
