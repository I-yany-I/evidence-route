"""Small in-memory remotezip substitute used by preparation tests."""

from __future__ import annotations

import binascii
import io
from dataclasses import dataclass


@dataclass
class FakeZipInfo:
    filename: str
    file_size: int
    CRC: int
    compress_size: int


class FakeRemoteZip:
    def __init__(self) -> None:
        self._members: list[tuple[FakeZipInfo, bytes]] = []
        self.extract_calls = 0
        self.extractall_calls = 0
        self.open_calls = 0

    def add(
        self,
        name: str,
        payload: bytes = b"",
        *,
        file_size: int | None = None,
        compress_size: int | None = None,
        crc: int | None = None,
    ) -> FakeZipInfo:
        info = FakeZipInfo(
            filename=name,
            file_size=len(payload) if file_size is None else file_size,
            CRC=(binascii.crc32(payload) & 0xFFFFFFFF) if crc is None else crc,
            compress_size=len(payload) if compress_size is None else compress_size,
        )
        self._members.append((info, payload))
        return info

    def infolist(self) -> list[FakeZipInfo]:
        return [info for info, _ in self._members]

    def open(self, name: str | FakeZipInfo, mode: str = "r") -> io.BytesIO:
        if mode != "r":
            raise ValueError("FakeRemoteZip supports read mode only")
        self.open_calls += 1
        for info, payload in self._members:
            if info is name or info.filename == name:
                return io.BytesIO(payload)
        raise KeyError(name)

    def extract(self, *args, **kwargs):
        del args, kwargs
        self.extract_calls += 1
        raise AssertionError("extract must not be used")

    def extractall(self, *args, **kwargs):
        del args, kwargs
        self.extractall_calls += 1
        raise AssertionError("extractall must not be used")

    def close(self) -> None:
        return None

    def __enter__(self) -> FakeRemoteZip:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
