import zipfile
import pickle
import os
import time
from contextlib import contextmanager

import numpy as np
import scipy


@contextmanager
def store(filename, *args, enable=True, restore=False, **kwargs):
    """
    Contextmanager which yields a `CompressedStore` in write/append mode.
    """
    if enable:
        inner = CompressedStore(filename, *args, restore=restore, **kwargs)
        try:
            yield inner
        finally:
            inner.close()
    else:
        yield None


def find_i(namelist):
    x = -1
    for s in namelist:
        try:
            x = max(x, int(s))
        except ValueError:
            pass
    return x+1


class CompressedStore:
    """
    A compressed, progressively writable, object store. Writes individual objects as files in a zip archive.
    Can be read lazily and iterated/indexed as if it were a container.

    All writes will be immediately persisted on disk, so programme interrupts should not result in loss of data.
    """

    def __new__(self, filename, *args, restore=False, read=False, **kwargs):
        if read is False and restore is False:
            return CompressedStoreV2(
                filename, *args, restore=restore, read=read, **kwargs
            )

        z = zipfile.ZipFile(filename, "r", compression=zipfile.ZIP_DEFLATED)
        try:
            version = int(z.read("version"))
        except KeyError:
            return CompressedStoreV1(
                filename, *args, restore=restore, read=read, **kwargs
            )
        if version == 2:
            return CompressedStoreV2(
                filename, *args, restore=restore, read=read, **kwargs
            )
        else:
            raise Exception(f"Unknown store version {version}")
        z.close()


class CompressedStoreV2:
    """
    A compressed, progressively writable, object store. Writes individual objects as files in a zip archive.
    Can be read lazily and iterated/indexed as if it were a container.

    All writes will be immediately persisted on disk, so programme interrupts should not result in loss of data.
    """

    version = 2

    def __init__(
        self,
        filename,
        # References to these are held once per store. Each classifier then keeps indexes
        # into these constant sets
        initial_X_labelled=None,
        initial_X_unlabelled=None,
        initial_Y_labelled=None,
        initial_Y_oracle=None,
        restore=False,
        read=False,
    ):
        if read:
            mode = "r"
        elif restore:
            mode = "a"
        else:
            mode = "w"
        self.filename = filename
        self.mode = mode

        self.initial_X_labelled = initial_X_labelled
        self.initial_X_unlabelled = initial_X_unlabelled
        self.initial_Y_labelled = initial_Y_labelled
        self.initial_Y_oracle = initial_Y_oracle

        while os.path.exists(filename + ".lock"):
            print(f"Blocking on store lock for file {filename}")
            time.sleep(10)

        try:
            self.zip = zipfile.ZipFile(
                self.filename, mode, compression=zipfile.ZIP_DEFLATED
            )
        except Exception as e:
            print(f"Failed to open compressed store {filename}")
            raise e

        if mode == "w":
            self.zip.writestr("version", "2")
            with self.zip.open("arrays.npz", "w", force_zip64=True) as filelike:
                savez(
                    filelike,
                    initial_X_labelled=initial_X_labelled,
                    initial_X_unlabelled=initial_X_unlabelled,
                    initial_Y_labelled=initial_Y_labelled,
                    initial_Y_oracle=initial_Y_oracle,
                )
        else:
            with self.zip.open("arrays.npz") as filelike:
                arrays = loadz(filelike)
                if initial_X_labelled is not None:
                    assert array_equal(initial_X_labelled, arrays["initial_X_labelled"])
                if initial_X_unlabelled is not None:
                    assert array_equal(
                        initial_X_unlabelled, arrays["initial_X_unlabelled"]
                    )
                if initial_Y_labelled is not None:
                    assert array_equal(initial_Y_labelled, arrays["initial_Y_labelled"])
                if initial_Y_oracle is not None:
                    assert array_equal(initial_Y_oracle, arrays["initial_Y_oracle"])
                self.initial_X_labelled = arrays["initial_X_labelled"]
                self.initial_X_unlabelled = arrays["initial_X_unlabelled"]
                self.initial_Y_labelled = arrays["initial_Y_labelled"]
                self.initial_Y_oracle = arrays["initial_Y_oracle"]

        self.i = find_i(self.zip.namelist())

    def append(self, obj):
        self.zip.writestr(str(self.i), pickle.dumps(obj))
        assert (
            find_i(self.zip.namelist()) == self.i + 1
        ), f"{find_i(self.zip.namelist())} == {self.i + 1}\n{self.zip.namelist()}"
        # To survive unexpected interrupts (from OS, not exceptions) we need to write the zip, then re-open it in append mode.
        # Otherwise changes will be lost because the finalizer doesn't run.
        self.zip.close()
        self.zip = zipfile.ZipFile(self.filename, "a", compression=zipfile.ZIP_DEFLATED)
        self.i += 1

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
                    self.set_pools(pickle.Unpickler(self.zip.open(str(x))).load())
                    for x in range(i.start, i.stop or self.i, i.step or 1)
                ]
            except KeyError:
                raise IndexError(f"index {i} out of range for store of length {self.i}")

        if i < 0:
            # print(f"Negative index {i} for len {self.i} gives a filename of {self.i+i}")
            i = self.i + i
        try:
            return self.set_pools(pickle.Unpickler(self.zip.open(str(i))).load())
        except KeyError:
            raise IndexError(f"index {i} out of range for store of length {self.i}")

    def set_pools(self, item):
        if not hasattr(item, "initial_X_labelled"):
            item.initial_X_labelled = self.initial_X_labelled
        if not hasattr(item, "initial_X_unlabelled"):
            item.initial_X_unlabelled = self.initial_X_unlabelled
        if not hasattr(item, "initial_Y_labelled"):
            item.initial_Y_labelled = self.initial_Y_labelled
        if not hasattr(item, "initial_Y_oracle"):
            item.initial_Y_oracle = self.initial_Y_oracle

        return item

    def indexes(self):
        return self.zip.namelist()

    # As written this is a single use iterable, create a closure here which keeps an ephemeral counter.
    def __iter__(self):
        i = 0
        while i <= self.i:
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
        self.i = find_i(self.zip.namelist())


class CompressedStoreV1:
    """
    A compressed, progressively writable, object store. Writes individual objects as files in a zip archive.
    Can be read lazily and iterated/indexed as if it were a container.

    All writes will be immediately persisted on disk, so programme interrupts should not result in loss of data.
    """

    version = 1

    def __init__(self, filename, *args, restore=False, read=False, **kwargs):
        if read:
            mode = "r"
        elif restore:
            mode = "a"
        else:
            mode = "w"
        self.filename = filename
        self.mode = mode

        while os.path.exists(filename + ".lock"):
            print(f"Blocking on store lock for file {filename}")
            time.sleep(10)

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
            # print(f"Negative index {i} for len {self.i} gives a filename of {self.i+i}")
            i = self.i + i
        try:
            return pickle.Unpickler(self.zip.open(str(i))).load()
        except KeyError:
            raise IndexError(f"index {i} out of range for store of length {self.i}")

    def indexes(self):
        return self.zip.namelist()

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


def savez(filelike, **arrays):
    args = {}
    for name, array in arrays.items():
        if isinstance(array, scipy.sparse.csr_matrix):
            args[name + "@data"] = (array.data,)
            args[name + "@indices"] = (array.indices,)
            args[name + "@indptr"] = (array.indptr,)
            args[name + "@shape"] = (array.shape,)
        else:
            args[name] = array
    return np.savez(filelike, **args)


def loadz(filelike):
    arrays = np.load(filelike, allow_pickle=True)
    out = {}
    for name, array in arrays.items():
        splits = name.split("@")
        if len(splits) == 2:

            out[splits[0]] = scipy.sparse.csr_matrix(
                (
                    arrays[splits[0] + "@data"][0],
                    arrays[splits[0] + "@indices"][0],
                    arrays[splits[0] + "@indptr"][0],
                ),
                shape=arrays[splits[0] + "@shape"][0],
            )
        else:
            out[name] = array
    return out

def array_equal(a, b):
    if isinstance(a, scipy.sparse.csr_matrix):
        return (a!=b).nnz == 0
    return np.array_equal(a, b)
