from dataclasses import dataclass
from typing import Dict, Any, Callable

import pandas as pd
from pprint import pprint, pformat


@dataclass
class Config:
    dataset_name: str
    method_name: str
    dataset_mutator_name: str
    model_name: str
    meta: Dict["str", Any]
    method: Callable = None
    dataset: Callable = None
    dataset_mutator: Callable = None

    def serialize(self):
        meta_str = "__".join(
            [
                f"{k}={v}" if k != "stop_function" else f"{k}={v[0]}"
                for k, v in self.meta.items()
            ]
        )
        return f"{self.dataset_name}__{self.dataset_mutator_name}__{self.method_name}__{self.model_name}__{meta_str}"

    def json(self):
        return {
            "dataset_name": self.dataset_name,
            "method_name": self.method_name,
            "dataset_mutator_name": self.dataset_mutator_name,
            "model_name": self.model_name,
            "meta": {
                k: v if k != "stop_function" else v[0] for k, v in self.meta.items()
            },
        }

    def __repr__(self):
        return pformat(self.json())


class Configurations:
    def __init__(self, matrix):
        self.configurations = []
        self.meta = matrix["meta"]

        for dataset in matrix["datasets"]:
            for method in matrix["methods"]:
                for model in matrix["models"]:
                    for dataset_mutator in matrix["dataset_mutators"].items():
                        self.configurations.append(
                            Config(
                                dataset_name=dataset[0],
                                dataset=dataset[1],
                                method_name=method[0],
                                method=method[1],
                                dataset_mutator_name=dataset_mutator[0],
                                dataset_mutator=dataset_mutator[1],
                                model_name=model,
                                meta=matrix["meta"],
                            )
                        )

    def __repr__(self):
        return pformat(
            {
                "meta": self.meta.__repr__(),
                "configurations": self.configurations.__repr__(),
            }
        )

    def __iter__(self, *args, **kwargs):
        return self.configurations.__iter__(*args, **kwargs)

    def __getitem__(self, *args, **kwargs):
        return self.configurations.__getitem__(*args, **kwargs)

    def __len__(self):
        return len(self.configurations)
