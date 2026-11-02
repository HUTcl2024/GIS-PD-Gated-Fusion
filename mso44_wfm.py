"""Read the single-frame INT16 :WFM#003 YT files verified in this project.

Requires numpy. No CSV conversion; raw values are memory mapped read-only.
Reference: https://download.tek.com/manual/077022005Web_1.pdf
Other encodings / FastFrame / envelope waveforms deliberately fail closed.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import struct
import numpy as np


@dataclass(frozen=True)
class WfmInfo:
    path: str
    samples: int
    dt_s: float
    t0_s: float
    y_scale: float
    y_offset: float
    y_unit: str
    x_unit: str
    data_offset: int
    curve_end: int
    byte_order: str

    @property
    def sample_rate_hz(self):
        return 1.0 / self.dt_s


class Wfm:
    def __init__(self, path):
        path = Path(path).resolve()
        with path.open('rb') as stream:
            h = stream.read(838)
        if len(h) < 838 or h[2:10] != b':WFM#003':
            raise ValueError(f'Unsupported WFM version: {path}')
        endian = {b'\x0f\x0f': '<', b'\xf0\xf0': '>'}.get(h[:2])
        if endian is None:
            raise ValueError('Unknown byte order')
        def unpack(fmt, offset):
            return struct.unpack_from(endian + fmt, h, offset)[0]
        if (unpack('I', 72) != 0 or unpack('I', 78) != 0
                or unpack('I', 82) != 1 or unpack('I', 110) != 1):
            raise ValueError('Only single-frame waveforms supported')
        if (unpack('I', 114), unpack('I', 118), unpack('I', 122)) != (1, 1, 2):
            raise ValueError('Only ordinary analog YT vectors supported')
        if h[15] != 2 or unpack('I', 240) != 0 or unpack('I', 244) != 0:
            raise ValueError('Only INT16 SAMPLE storage supported')
        # V003: +2 bytes after offset 154; +4 for each point-density field.
        curve = unpack('I', 16)
        pre, start, stop, post, end = [unpack('I', o) for o in (818, 822, 826, 830, 834)]
        if not (838 <= curve and 0 <= pre <= start < stop <= post <= end
                and all(v % 2 == 0 for v in (pre, start, stop, post, end))
                and curve + end + 8 <= path.stat().st_size):
            raise ValueError('Invalid curve offsets or truncated file')
        n = (stop - start) // 2
        if unpack('I', 504) != (post - pre) // 2:
            raise ValueError('Record length disagrees with curve offsets')
        dt, t0 = unpack('d', 488), unpack('d', 496)
        scale, offset = unpack('d', 168), unpack('d', 176)
        if not all(np.isfinite(v) for v in (dt, t0, scale, offset)) or dt <= 0 or scale == 0:
            raise ValueError('Invalid axis scaling')
        unit = lambda o: h[o:o+20].split(b'\0')[0].decode('ascii')
        if unit(508) != 's':
            raise ValueError('Time axis is not in seconds')
        self.info = WfmInfo(str(path), n, dt, t0 + start // 2 * dt,
                            scale, offset, unit(188), unit(508), curve + start,
                            curve + end, endian)
        self.raw = np.memmap(path, dtype=endian+'i2', mode='r',
                             offset=self.info.data_offset, shape=(n,))

    def values(self, start=0, stop=None, step=1):
        """Values in the file's axis units (V here); no CT A/pC calibration implied."""
        return self.raw[start:stop:step].astype(np.float64) * self.info.y_scale + self.info.y_offset

    def times(self, start=0, stop=None, step=1):
        if stop is None:
            stop = self.info.samples
        return self.info.t0_s + np.arange(start, stop, step, dtype=np.float64) * self.info.dt_s

    def verify_checksum(self):
        """Tektronix unsigned byte sum through the end of the curve buffer."""
        total = 0
        with open(self.info.path, 'rb') as f:
            left = self.info.curve_end
            while left:
                chunk = f.read(min(left, 1024*1024))
                if not chunk:
                    raise ValueError('Truncated file')
                total += int(np.frombuffer(chunk, dtype=np.uint8).sum(dtype=np.uint64))
                left -= len(chunk)
            stored = struct.unpack(self.info.byte_order+'Q', f.read(8))[0]
        return total == stored


def read_triplet(ch1, ch2, ch3):
    """CH1=UHF, CH2=phase reference, CH3=HFCT. Check grids, not sensor delays."""
    waves = [Wfm(p) for p in (ch1, ch2, ch3)]
    ref = waves[0].info
    for w in waves[1:]:
        i = w.info
        if (i.samples != ref.samples or not np.isclose(i.dt_s, ref.dt_s, rtol=1e-10, atol=0)
                or abs(i.t0_s-ref.t0_s) > ref.dt_s * 1e-5):
            raise ValueError('Three channels have different time grids')
    return dict(zip(('uhf', 'phase_ref', 'hfct'), waves))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path)
    args = parser.parse_args()
    for path in args.files:
        w = Wfm(path)
        print(json.dumps({**asdict(w.info), 'checksum_ok': w.verify_checksum()}, ensure_ascii=False))
