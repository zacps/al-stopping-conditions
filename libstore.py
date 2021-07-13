import zipfile
import pickle
from contextlib import contextmanager


@contextmanager
def store(filename, enable, restore=False):
    """
    Contextmanager which yields a `CompressedStore` in write/append mode.
    """
    if enable:
        inner = CompressedStore(filename, restore=restore)
        try:
            yield inner
        finally:
            inner.close()
    else:
        yield None


class CompressedStore:
    """
    A compressed, progressively writable, object store. Writes individual objects as files in a zip archive.
    Can be read lazily and iterated/indexed as if it were a container.

    All writes will be immediately persisted on disk, so programme interrupts should not result in loss of data.
    """

    def __init__(self, filename, restore=False, read=False):
        if read:
            mode = "r"
        elif restore:
            mode = "a"
        else:
            mode = "w"
        self.filename = filename
        self.mode = mode
        try:
            self.zip = zipfile.ZipFile(
                self.filename, mode, compression=zipfile.ZIP_DEFLATED
            )
        except Exception as e:
            print(f"Failed to open compressed store {filename}")
            raise e
        self.i = len(self.zip.namelist())

    def append(self, obj):
        self.zip.writestr(str(self.i), pickle.dumps(obj))
        assert len(self.zip.namelist()) == self.i + 1
        # To survive unexpected interrupts (from OS, not exceptions) we need to write the zip, then re-open it in append mode.
        # Otherwise changes will be lost because the finalizer doesn't run.
        self.zip.close()
        self.zip = zipfile.ZipFile(self.filename, "a", compression=zipfile.ZIP_DEFLATED)
        self.i = len(self.zip.namelist())

    def __len__(self):
        return self.i

    def __getitem__(self, i):
        if isinstance(i, slice):
            if i.start < 0:
                i.start = self.i - i.start
            if i.stop is not None and i.stop < 0:
                i.stop = self.i - i.stop
            try:
                return [
                    pickle.Unpickler(self.zip.open(str(x))).load()
                    for x in range(i.start, i.stop or self.i, i.step or 1)
                ]
            except KeyError:
                raise IndexError(f"index {i} out of range for store of length {self.i}")

        if i < 0:
            i = self.i - 1 - i
        try:
            return pickle.Unpickler(self.zip.open(str(i))).load()
        except KeyError:
            raise IndexError(f"index {i} out of range for store of length {self.i}")

    # As written this is a single use iterable, create a closure here which keeps an ephemeral counter.
    def __iter__(self):
        i = 0
        while i < self.i:
            yield self[i]
            i += 1

    def close(self):
        self.zip.close()

    def __getstate__(self):
        if hasattr(self, "zip"):
            odict = self.__dict__.copy()
            del odict["zip"]
        return odict

    def __setstate__(self, d):
        self.__dict__ = d
        self.zip = zipfile.ZipFile(
            self.filename,
            "r" if self.mode == "r" else "a",
            compression=zipfile.ZIP_DEFLATED,
        )
        self.i = len(self.zip.namelist())
