import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ImagePath:
    value: str

    _pattern = re.compile(r"\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)

    def __post_init__(self):
        if not self._pattern.search(self.value):
            raise ValueError("画像ファイルのパスを入力してください")

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"ImagePath({self.value!r})"
