from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple, Any


class Source(ABC):

    @abstractmethod
    def load(self) -> Iterator[Tuple[List[List[str]], Any]]:
        raise NotImplementedError
