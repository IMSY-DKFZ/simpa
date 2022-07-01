# SPDX-FileCopyrightText: 2021 Division of Intelligent Medical Systems, DKFZ
# SPDX-FileCopyrightText: 2021 Janek Groehl
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod


class SerializableSIMPAClass(ABC):
    """
    TODO
    """

    @abstractmethod
    def serialize(self) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def deserialize(dictionary_to_deserialize: dict):
        pass
