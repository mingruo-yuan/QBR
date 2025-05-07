import pandas as pd
import json
import os
import jsonlines

def write_to_file(data, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))
def read_from_json(path_of_file):
    with open(path_of_file, 'r' ,encoding='utf-8') as f:
        results = json.load(f)
    return results

def read_one_file_scope(path):
    df = pd.read_excel(io=path, sheet_name=None)
    print("There are ", len(df.keys()), " pages in this sheet")  # Number of pages

    for each_page in df:  # For each page
        document = {}
        document["_id"] = f"CLIC-{each_page}"
        document["title"] = ""
        document["text"] = ""

        for idx, data in df[each_page].iterrows():
            if (type(data['Text_ID']) == str) and (data['Text_ID'] != "S000"):
                if data['Text'].startswith("Paragraph"):
                    document["text"] = document["text"] + data['Text'][13:] + " "
                else:
                    document["text"] = document["text"] + data['Text'] + " "
            if (type(data['Text_ID']) == str) and (data['Text_ID'] == "S000"):
                document["title"] = data['Text']
        with jsonlines.open('dataset/corpus.jsonl', mode='a') as writer:
            writer.write(document)

    return True

if __name__ == '__main__':
    ## For user input/ generated user story (!)
    b = read_from_json("Medical/test.json")
    for i in b:
        if i["story"].find(":") != -1:
            story_query = {}
            page = i["page"]
            index = i["index"]
            story_query["_id"] = f"CLIC-{page}-{index}"
            story_query["text"] = i["story"][i["story"].find(":") + 2:].strip()
        else:
            story_query = {}
            page = i["page"]
            index = i["index"]
            story_query["_id"] = f"CLIC-{page}-{index}"
            story_query["text"] = i["story"].strip()
        with jsonlines.open('data/medical/queries.jsonl', mode='a') as writer:
            writer.write(story_query)

    # for i in b:
    #     if i["Question"].find(".") != -1:
    #         story_query = {}
    #         page = i["page"]
    #         index = i["index"]
    #         story_query["_id"] = f"CLIC-{page}-{index}"
    #         story_query["text"] = i["Question"][i["Question"].find(".") + 1:].strip()
    #     elif i["Question"].find("-") != -1:
    #         story_query = {}
    #         page = i["page"]
    #         index = i["index"]
    #         story_query["_id"] = f"CLIC-{page}-{index}"
    #         story_query["text"] = i["Question"][i["Question"].find("-") + 1:].strip()
    #     else:
    #         story_query = {}
    #         page = i["page"]
    #         index = i["index"]
    #         story_query["_id"] = f"CLIC-{page}-{index}"
    #         story_query["text"] = i["Question"].strip()
    #     with jsonlines.open('data/medical/queries.jsonl', mode='a') as writer:
    #         writer.write(story_query)

    # ## string+scope
    # sample_500 = read_from_json("Medical/questions_w_section.json")
    # scope_all = read_from_json("Medical/scope_medical_result_all.json")
    #
    # for i in sample_500:
    #     question_string_scope = {}
    #     scope_list = []
    #     page = i["page"]
    #     index = i["index"]
    #     question_string_scope["_id"] = f"CLIC-{page}-{index}"  # page-index
    #
    #     # Augmentated String-Scope
    #     # if i["Identifier"] in augment_dict.keys():
    #     #     question_string_scope["title"] = augment_dict[i["Identifier"]]
    #     # else:
    #     #     question_string_scope["title"] = i["Question"]
    #
    #     # Original String-Scope
    #     question_string_scope["title"] = i["Question"]
    #
    #     for j in i["Answered in text?"]:
    #         scope_list.append(scope_all[j])
    #         temp_scope = "\n".join(scope_list)
    #     question_string_scope["text"] = temp_scope
    #
    #     question_string_scope["metadata"] = i["Answered in text?"]
    #
    #     with jsonlines.open('data/medical/medical_seciton/corpus.jsonl', mode='a') as writer:
    #         writer.write(question_string_scope)

    # ## document
    # document_all = read_from_json("Medical/batch_new.json")
    #
    # for i in document_all:
    #     document = {}
    #     page = str(i)
    #     document["_id"] = f"CLIC-{page}"  # page
    #     document["title"] = document_all[i]["text"][0]
    #     document["text"] = "\n".join(document_all[i]["text"][1:])
    #
    #     with jsonlines.open('data/medical_document/corpus.jsonl', mode='a') as writer:
    #         writer.write(document)