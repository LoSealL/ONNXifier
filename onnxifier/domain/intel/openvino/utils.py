"""
Copyright (C) 2024 The ONNXIFIER Authors.

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

from contextlib import suppress
from typing import List, Literal, Optional, overload


def text_to_boolean(text: str | Literal["0", "1", "false", "true"]) -> bool:
    """Convert common text to bool

    Args:
        text (str): integer string or false/true literals
    """

    with suppress(ValueError):
        integer = int(text)
        return bool(integer)
    return text.lower() == "true"


@overload
def text_to_integers(text: None) -> None:
    """Convert comma-separated text to list of integers"""


@overload
def text_to_integers(text: str) -> List[int]:
    """Convert comma-separated text to list of integers"""


def text_to_integers(text: Optional[str]) -> List[int] | None:
    """Convert comma-separated text to list of integers

    Args:
        text (str | None): comma-separated integers
    """

    if text is None:
        return None
    elif text.strip() == "":
        return []
    return list(map(int, text.split(",")))
