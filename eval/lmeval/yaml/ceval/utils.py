import re
import datasets



def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):

        out_doc = {
            "question": doc["question"],
            "A": doc["A"],
            "B": doc["B"],
            "C": doc["C"],
            "D": doc["D"],
            "choice": ["A"+doc["A"], "B"+doc["B"], "C"+doc["C"], "D"+doc["D"]],
            "answer": doc["answer"]
        }
        return out_doc

    return dataset.map(_process_doc)
