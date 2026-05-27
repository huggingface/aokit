"""
"""

from pathlib import Path
import mmap
import struct


ELF_HEADER = b'\x7fELF\x02\x01'
PT_GNU_STACK = 0x6474E551
PF_X = 0x1

OFFSET_PROGRAM_HEADER_OFFSET = 32
OFFSET_PROGRAM_HEADER_ESIZE = 54
OFFSET_PROGRAM_HEADER_COUNT = 56
OFFSET_PROGRAM_HEADER_FLAGS = 4


def clear_execstack(path: Path) -> bool:
    with path.open('r+b') as file:
        buffer = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_WRITE)
        assert buffer[:len(ELF_HEADER)] == ELF_HEADER
        ph_offset: int = struct.unpack_from('Q', buffer=buffer, offset=OFFSET_PROGRAM_HEADER_OFFSET)[0]
        ph_esize: int = struct.unpack_from('H', buffer=buffer, offset=OFFSET_PROGRAM_HEADER_ESIZE)[0]
        ph_count: int = struct.unpack_from('H', buffer=buffer, offset=OFFSET_PROGRAM_HEADER_COUNT)[0]
        for index in range(ph_count):
            offset = ph_offset + index * ph_esize
            flags_name, flags_value = struct.unpack_from('II', buffer=buffer, offset=offset)
            if flags_name != PT_GNU_STACK:
                continue
            if flags_value & PF_X == 1:
                write_offset = offset + OFFSET_PROGRAM_HEADER_FLAGS
                write_value = flags_value & ~PF_X
                struct.pack_into('I', buffer, write_offset, write_value)
                return True
            return False # pragma: no cover
        return False # pragma: no cover
