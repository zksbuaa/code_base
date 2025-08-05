import re
import datasets



def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):

        out_doc = {
            "question": doc["question"],
            "choices": doc["choices"],
            "choice": ["A."+doc["choices"][0], "B."+doc["choices"][1], "C."+doc["choices"][2], "D."+doc["choices"][3]],
            "answer": doc["answer"]
        }
        return out_doc

    return dataset.map(_process_doc)
