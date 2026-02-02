import pickle
import json
from ast import literal_eval
from goatools.obo_parser import GODag



def cafa_f1(answer_closure: set, prediction_closure: set):
    # True positives
    tp = len(prediction_closure & answer_closure)
    # False positives
    fp = len(prediction_closure - answer_closure)
    # False negatives
    fn = len(answer_closure - prediction_closure)

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def closure(term_set, go_dag):
    
    GO_ROOTS = {"GO:0008150", "GO:0003674", "GO:0005575"}

    closed = set()
    for t in term_set:
        if t not in go_dag:
            continue
        closed.add(t)
        closed |= go_dag[t].get_all_parents()  
    closed -= GO_ROOTS
    return closed

def calculate_go(answer, prediction):
    
    go_dag = GODag("./go-basic.obo")
    with open("./go_dict.json", "r") as f:
        go_id_dict = json.load(f)
    with open("./go_name_to_id.pkl", "rb") as f:
        name_to_id = pickle.load(f)
        
    prediction = literal_eval(prediction)
    
    # get answer go id list
    answer_list = []
    for answer_dict in answer:
        answer_go = answer_dict["go"].replace("_", " ")
        answer_evidence = answer_dict["evidence"]
        if answer_go.startswith("GO"):
            answer_go = answer_go.split(" ")[0]
            answer_list.append(answer_go)
        else:
            answer_go_id = go_id_dict[answer_evidence][answer_go]
            answer_list.append(answer_go_id)
        
    # get answer closure
    answer_closure = set()
    for go_id in answer_list:
        answer_closure.add(go_id)
        answer_closure.update(closure({go_id}, go_dag))
        
        
    # get prediction go id list
    prefixes_to_remove = [
            "involved in ", "located in ", "part of ", "enables ",
            "interacts with ", "binds to ", "encodes ", "activated by ",
            "regulates ", "binds ", "guides ", "functions as ",
            "participates in ", "facilitates ", "contains ", "regulated by ",
            "metabolizes "
        ]
    
    hallucination = 0
    count = 0
    prediction_list = []
    for prediction_dict in prediction:
        prediction_go = prediction_dict["go"]
        for prefix in prefixes_to_remove:
            if prediction_go.startswith(prefix):
                prediction_go = prediction_go[len(prefix):]
        if name_to_id.get(prediction_go, None) is not None :
            prediction_list.append(name_to_id.get(prediction_go))
            count += 1
        elif prediction_go.startswith("GO"):
            prediction_list.append(prediction_go.split(" ")[0])
            count +=1
        else:
            hallucination += 1
            
        
    # get prediction closure
    prediction_closure = set()
    for go_id in prediction_list:
        prediction_closure.add(go_id)
        prediction_closure.update(closure({go_id}, go_dag))
        
        
    # calculate CAFA
    precision, recall, f1 = cafa_f1(answer_closure, prediction_closure)
    if count+hallucination > 0:
        hall_rate = hallucination/(count+hallucination)
    else:
        hall_rate = 0.0
    return precision, recall, f1, hall_rate
    
