"""Shared generator base model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

OutputT = TypeVar("OutputT")

PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]


class DataGenerator(BaseModel, ABC, Generic[OutputT]):
    """Pydantic base class for table-generating models."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, validate_default=True)

    _yielded: bool = PrivateAttr(default=False)

    @abstractmethod
    def generate(self) -> OutputT:
        """Generate and return the synthetic dataset."""

    def __iter__(self) -> DataGenerator[OutputT]:
        self._yielded = False
        return self

    def __next__(self) -> OutputT:
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.generate()
