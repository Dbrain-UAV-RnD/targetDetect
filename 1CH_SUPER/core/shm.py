import pickle
import struct
import numpy as np
from multiprocessing import shared_memory

_HDR = struct.Struct("<IdQI")


class _Slot:
    def __init__(self, name, size, create):
        self.name = name
        if create:
            try:
                old = shared_memory.SharedMemory(name=name)
                old.close()
                old.unlink()
            except FileNotFoundError:
                pass
            self.shm = shared_memory.SharedMemory(name=name, create=True,
                                                  size=_HDR.size + size)
            self.shm.buf[:_HDR.size] = _HDR.pack(0, 0.0, 0, 0)
        else:
            self.shm = shared_memory.SharedMemory(name=name, track=False)
        self._owner = create

    def _write(self, payload, t, frame_id):
        seq, _, _, _ = _HDR.unpack_from(self.shm.buf, 0)
        seq += 1
        _HDR.pack_into(self.shm.buf, 0, seq, t, frame_id, len(payload))
        self.shm.buf[_HDR.size:_HDR.size + len(payload)] = payload
        _HDR.pack_into(self.shm.buf, 0, seq + 1, t, frame_id, len(payload))

    def _read(self, retries=4):
        for _ in range(retries):
            seq0, t, fid, n = _HDR.unpack_from(self.shm.buf, 0)
            if seq0 == 0 or seq0 & 1 or n == 0:
                continue
            payload = bytes(self.shm.buf[_HDR.size:_HDR.size + n])
            seq1, _, _, _ = _HDR.unpack_from(self.shm.buf, 0)
            if seq0 == seq1:
                return payload, t, fid, seq0
        return None, 0.0, 0, 0

    def close(self):
        self.shm.close()


class FrameSlot(_Slot):

    def __init__(self, name, shape, dtype=np.uint8, create=False):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.nbytes = int(np.prod(self.shape)) * self.dtype.itemsize
        super().__init__(name, self.nbytes, create)

    def write(self, img, t, frame_id):
        assert img.nbytes == self.nbytes, (img.shape, self.shape)
        self._write(img.tobytes(), t, frame_id)

    def read(self):
        payload, t, fid, seq = self._read()
        if payload is None:
            return None, 0.0, 0, 0
        img = np.frombuffer(payload, self.dtype).reshape(self.shape)
        return img, t, fid, seq


class BlobSlot(_Slot):
    def __init__(self, name, size=1 << 20, create=False):
        super().__init__(name, size, create)

    def write(self, obj, t, frame_id=0):
        self._write(pickle.dumps(obj, protocol=4), t, frame_id)

    def read(self):
        payload, t, fid, seq = self._read()
        if payload is None:
            return None, 0.0, 0, 0
        return pickle.loads(payload), t, fid, seq
