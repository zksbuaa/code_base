def doc_to_text(doc):
    return ""


def doc_to_target(doc):
    return int(doc['answer']) - 1


def doc_to_choice(doc):
    return [doc['sentence'].replace("_", doc['option1']), doc['sentence'].replace("_", doc['option2'])]
