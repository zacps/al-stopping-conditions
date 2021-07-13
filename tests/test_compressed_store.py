import pytest

from tempfile import NamedTemporaryFile

from libstore import CompressedStore

@pytest.mark.skip('named temporary file holds an fp which blocks opening on windows')
def test_compressed_store():
    with NamedTemporaryFile('ab') as file:
        store = CompressedStore(file.name)
        
        for i in range(10):
            store.write(i)
            
        store.close(); del store
        
        store = CompressedStore(file.name, read=True)
        assert list(store) == list(range(10))