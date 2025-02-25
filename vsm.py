import os
import sys
import time
import json
import subprocess
import multiprocessing as mp
import xml.etree.ElementTree as ET

from tqdm import tqdm
from hashlib import md5
from stopwords import STOPWORDS
from dateutil.parser import parse

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser
from gensim.similarities import SparseMatrixSimilarity
from gensim.parsing.preprocessing import preprocess_string

from multiprocessing import Manager
manager = Manager()
q_to_store = manager.Queue()


def text_processor(docs):
    processing_text = [word for word in preprocess_string(docs) if not word in STOPWORDS and len(word)>3]
    return processing_text


def bug_reader(bug_report_path, code_base_path):
    bugs = []

    tree = ET.parse(bug_report_path)
    root = tree.getroot()
   
    for element in root.findall(".//table"):
        bug_id = element[1].text
        summary = element[2].text or ''
        description = element[3].text or ''
        # print(bug_id, ',', summary, ',', description)
        bug_content = text_processor(summary + description)
        fixed_commit_time = element[8].text
        fixed_files = element[9].text.split('.java')
        fixed_files = [(file + '.java').strip() for file in fixed_files[:-1]]

        list_of_fixed_files = []
        for file_path in fixed_files:
            list_of_fixed_files.append(os.path.join(code_base_path, file_path))

        buggy_commit = element[10].text
        buggy_commit_time = element[11].text
        
        bug_data = {"id": bug_id,
                    "content": bug_content,
                    "buggy_commit": buggy_commit,
                    "buggy_commit_time": buggy_commit_time,
                    "fixed_commit_time": fixed_commit_time, 
                    "fixed_files": list_of_fixed_files}
        bugs.append(bug_data)
        print(list_of_fixed_files)
        
    bugs = sorted(bugs, key=lambda d: d['fixed_commit_time'])

    length = len(bugs)
    print('total bugs', length)
    starting_index = length - int(length*0.4)
    new_bugs = bugs[starting_index:length]

    new_bugs = sorted(new_bugs, key=lambda d: d['buggy_commit_time'])
    
    return new_bugs

def error_handler(e):
    print(f"Error occurred: {e}")

def mp_code_reader(code_base_path, storage_path):

    # Language.build_library(
    #     os.path.join("./languages.so"),
    #     [
    #         '/home/tree-sitter-python'
    #     ]
    # )

    added_files, modified_files = filter_files(code_base_path, storage_path)
    if os.path.exists(os.path.join(storage_path, "code_data.json")):
        with open(os.path.join(storage_path, "code_data.json"), "r") as f:
            code_data = json.load(f)
    else:
        code_data = {}
    
    if len(added_files + modified_files):
        pool = mp.Pool(mp.cpu_count()-2)
        for code_file in added_files + modified_files:
            pool.apply_async(code_files_reader, args=(code_file,),error_callback=error_handler) #, callback=update_code_data
        pool.close()
        pool.join()

    while not q_to_store.empty():
        single_data = q_to_store.get()
        code_data.update(single_data)
    print("Added finished")
    
    with open(os.path.join(storage_path, "code_data.json"), "w") as f:
        json.dump(code_data, f)
    
    return code_data

def filter_files(code_base_path, storage_path):

    added_files = []
    deleted_files = []
    modified_files = []
    code_files = []
    
    dir_path = os.walk(code_base_path)
    for parent_dir, dir_name, file_names in dir_path:
        for file_name in file_names:
            # print(file_name)
            if file_name.split(".")[-1].strip() == "java":
                code_files.append(os.path.join(parent_dir, file_name))

    if os.path.exists(os.path.join(storage_path, "code_data.json")):
        with open(os.path.join(storage_path, "code_data.json"), "r+") as f:
            code_data = json.load(f)

            for file_path in list(code_data.keys()):
                if not file_path in code_files:
                    deleted_files.append(file_path)
                    del code_data[file_path]

            for code_file in code_files:
                if os.path.getsize(code_file) and not code_file in code_data.keys():
                    added_files.append(code_file)
                elif os.path.getsize(code_file):
                    with open(code_file, encoding="utf8", errors="ignore") as file:
                        code_cont = file.read()
                        md5_val = md5(code_cont.encode()).hexdigest()
                    
                    if not md5_val == code_data[code_file]["md5"]:
                        code_data[code_file].update({"md5": md5_val})
                        modified_files.append(code_file)

            f.seek(0)
            f.truncate(0)
            json.dump(code_data, f)
    else:
        for code_file in code_files:
            added_files.append(code_file)

    return added_files, modified_files


def code_files_reader(code_file):
    try:
        # print(code_file)
        code_data = {}
        with open(code_file, encoding="utf8", errors="ignore") as f:
            if os.path.getsize(code_file):
                code_cont = f.read()
                md5_val = md5(code_cont.encode()).hexdigest()
                cont = text_processor(code_parser(code_cont))
                code_data[code_file] = {"content": cont, "md5": md5_val}
                
        q_to_store.put(code_data)
    except Exception as e:
        print(e)


def code_parser(code_cont):
    try:
        parser = Parser()
        JAVA_LANGUAGE = Language(tsjava.language(), 'java')
        parser.set_language(JAVA_LANGUAGE)
        parsed_code = str.encode(code_cont)
        tree = parser.parse(parsed_code)
        code_lines = code_cont.split('\n')

        identifier = ""
        nodes = [tree.root_node]
        while nodes:
            temp = []
            for node in nodes:
                for child in node.children:
                    temp.append(child)
                    if(child.type == "identifier"):
                        identifier += code_lines[child.start_point[0]][child.start_point[1]:child.end_point[1]] + " "
            nodes = temp

        return code_cont + identifier
    
    except Exception as e:
        print(e)

def compute_similarity(text_data, bug_data):
    text_content = []
    file_names = []
    for file_name, content in text_data.items():
        text_content.append(content["content"])
        file_names.append(file_name)
    
    dct = Dictionary(text_content)

    code_vecs = [dct.doc2bow(text) for text in text_content]
    model = TfidfModel(code_vecs, "lfn")
    index = SparseMatrixSimilarity(model[code_vecs], num_features=len(dct))
    bug_content = bug_data["content"]
    sims = index[dct.doc2bow(bug_content)]
    sorted_files = sorted(range(len(sims)), key=lambda k: sims[k], reverse=True)

    results = []
    for i in sorted_files:
        results.append(file_names[i])

    return results[:500]

def evaluation(results, storage_path):
    
    map_value = 0
    ap_value = {}
    count = 0

    for bug_id, results in results.items():
        temp1 = 0
        ap_tmp = 0
        truth_num = 0
        file_paths = results["truth"]
        result = results["results"]

        if not len(file_paths) == 0:
            for file_path in file_paths:
                for i in result:
                    if i == file_path:
                        truth_num += 1
                                    
        if truth_num > 0:
            count += 1
        if not truth_num == 0:
            ap_value[bug_id] = {}
            for i, j in enumerate(result):
                if j in file_paths:
                    temp1 += 1
                    ap_tmp += temp1/(i+1.0)
            
            ap_value[bug_id]["AP"] = ap_tmp / len(file_paths)
    
    past_ap_value = {}
    if os.path.exists(os.path.join(storage_path, "evaluation.json")):
        with open(os.path.join(storage_path, "evaluation.json"), "r") as f:
            past_ap_value = json.load(f)
    past_ap_value.update(ap_value)
    
    with open(os.path.join(storage_path, "evaluation.json"), "w") as f:
        json.dump(past_ap_value, f)
    
    if not count == 0:
        for ap in ap_value.values():
            map_value += ap["AP"]
        map_value /= count
    else:
        map_value = 0

    print("The MAP is", map_value)


if __name__ == "__main__":

    proj = sys.argv[1]
    bug_report_path = sys.argv[2]
    code_base_path = sys.argv[3]
    storage_path = sys.argv[4]
    storage_path = os.path.join(f"{storage_path}", proj)
    if not os.path.exists(os.path.join(storage_path, "code/")):
        os.makedirs(os.path.join(storage_path, "code/"))
    
    results = {}
    start_time = time.time()
    print("read bug reports...")
    bug_data = bug_reader(bug_report_path, code_base_path)
    print("the time consuming is %f s" %(time.time() - start_time))

    for bug in tqdm(bug_data):
        print("processing {} now...".format(bug["id"]))
        cmd_1 = "cd " + code_base_path
        cmd_2 = "git reset --hard " + bug["buggy_commit"]
        # print("commit", bug["buggy_commit"])
        p = subprocess.Popen(cmd_1 + "&&" + cmd_2, stdout=subprocess.PIPE, shell=True)
        p.wait()

        start_time = time.time()
        print("read code files...")
        code_data = mp_code_reader(code_base_path, os.path.join(storage_path, "code/"))
        print("the time consuming is %f s" %(time.time() - start_time))

        start_time = time.time()
        print("compute similarities...")
        result = compute_similarity(code_data, bug)
        results[bug["id"]] = {"results": result, "truth": [fp for fp in bug["fixed_files"]]}
        print("the time consuming is %f s" %(time.time() - start_time))

        # evaluation(results, storage_path)
        # break

    with open(os.path.join(storage_path, "results.json"), "w") as f:
        json.dump(results, f)

