import json


def item_metadata_func(record: dict, seq_num: int, metadata: dict) -> dict:
    metadata["seq_num"] = seq_num
    return metadata


def format_docs(docs):
    docs_list = docs['context']
    return docs_list


def format_outputs(key,outputs):
    formatted_out = dict()
    formatted_out['question'] = outputs[key]['input']
    formatted_out['answer'] = outputs[key]['answer']
    print(formatted_out)
