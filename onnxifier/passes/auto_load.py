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

from importlib import import_module
from pathlib import Path
from typing import Optional, Set, Tuple

from ..logger import warning

TOP = Path(__file__).parent.parent.parent.resolve()


def auto_load(cwd: str | Path, filters: Optional[Set[str]] = None, top: Path = TOP):
    """Search and load python modules in `cwd`.

    Args:
        cwd (str | Path): module directory, glob recurrsively
        filters (Set[str], optional): patterns to filter out modules. Defaults to None.
        top (Path, optional): module top directory. Defaults to TOP.
    """
    models_dir = Path(cwd).resolve()
    package = models_dir.relative_to(top).as_posix().replace("/", ".")

    def _filter(p: Path) -> bool:
        pattern: Tuple[str, ...] = ("__",)
        if filters:
            pattern += tuple(filters)
        return all(i not in p.stem for i in pattern)

    for src in filter(_filter, models_dir.rglob("*.py")):
        module_url = src.relative_to(models_dir).with_suffix("").as_posix()
        try:
            import_module(f".{module_url.replace('/', '.')}", package)
        except ImportError as ex:
            warning(f"failed to load {module_url}: {ex}")
