import re

import datasets



def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        context = "\n".join(doc["context"])
        
        out_doc = {
            "context": context,
            "question": doc["question"],
            "choice": doc["choice"],
            "answer": doc["choice"].index(doc["answer"]),
        }
        return out_doc

    return dataset.map(_process_doc)
