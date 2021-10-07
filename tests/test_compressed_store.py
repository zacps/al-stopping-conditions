import os
import pytest

from tempfile import NamedTemporaryFile

from libstore import CompressedStore


@pytest.mark.skip("named temporary file holds an fp which blocks opening on windows")
def test_compressed_store():
    with NamedTemporaryFile("ab") as file:
        store = CompressedStore(file.name)

        for i in range(10):
            store.write(i)

        store.close()
        del store

        store = CompressedStore(file.name, read=True)
        assert list(store) == list(range(10))


def test_compressed_store_v2_indexing():
    with NamedTemporaryFile("ab", delete=False) as file:
        fname = file.name

    store = CompressedStore(fname)

    for i in range(10):
        store.append(i)

    store.close()
    del store

    items = list(range(10))
    store = CompressedStore(file.name, read=True)
    store.set_pools = lambda x: x

    assert list(store) == items
    assert store[0] == items[0]
    assert store[3] == items[3]
    assert store[-1] == items[-1]
    assert store[-2] == items[-2]
    assert store[2:7:2] == items[2:7:2]

    store.close()
    del store

    os.remove(fname)
