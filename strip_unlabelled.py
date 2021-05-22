import glob
import os
from libactive import CompressedStore

for name in glob.glob(f"{out_dir()}/classifiers/*"):
    if os.path.getsize(name) > 1024**3:
        store = CompressedStore(name, read=True)
        new_store = CompressedStore(name+".min")
        for clf in store:
            if hasattr(clf, X_unlabelled):
                del clf.X_unlabelled
            new_store.append(clf)
        store.close()
        new_store.close()
        os.unlink(name)
        os.rename(name+".min", name)