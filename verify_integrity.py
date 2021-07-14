import glob
import os
import zipfile

import numpy as np

from libactive import CompressedStore

# Resume at index
start_at = 0

paths = list(
    glob.glob(f"/home/zac/out/classifiers/*__none__uncertainty__random-forest__*.zip")
)
for i, name in enumerate(paths):
    if i < start_at:
        continue

    # Progress
    print(f"{i}/{len(paths)}")

    try:
        store = CompressedStore(name, read=True)
    except zipfile.BadZipFile as e:
        print("Bad zip file")
        print(e)
        continue

    if len(store) < 100:
        print("Run in progress, ignoring")
        continue

    if len(store) > 100:
        print(name)
        print(Exception(f"store has length {len(store)} but should have length 100"))
        continue

    try:
        for clf in store:
            assert hasattr(clf, "predict")
            assert hasattr(clf, "predict_proba")

            clf.predict_proba(np.ones((1, clf.X_training.shape[1])))

    except zipfile.BadZipFile as e:
        print(f"Failed to verify store {name}")
        print(e)
        continue
