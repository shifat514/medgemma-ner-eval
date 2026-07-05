from abc import ABC, abstractmethod
from typing import List

from datasets import DatasetDict


class BaseNERAdapter(ABC):
    def __init__(self, label_mode: str = "harmonized"):
        self.label_mode = label_mode

    @abstractmethod
    def load(self) -> DatasetDict:
        """Return DatasetDict with splits containing 'tokens' (list[str]) and 'bio' (list[str])."""

    @abstractmethod
    def label_list(self) -> List[str]:
        """Return the ordered list of BIO label strings for this adapter."""
