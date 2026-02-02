from ast import literal_eval
import pandas as pd


def calculate_expression(answer, prediction, category_weight=0.5, tissue_list_weight=0.5):
    """
    calculate gene expression metrics
    we consider tissue_list and category
    """
    
    if category_weight + tissue_list_weight != 1.0:
        raise ValueError("Weights must sum to 1.0.")
    
    
    predicted_data = literal_eval(prediction.replace('```json','').replace('```','').replace("</s>","").replace("<s>","").replace(".",""))
    true_data = answer

        
    # If predicted_data is not a dict, return 0.0
    if pd.isna(answer) or pd.isna(prediction):
        return 0.0


    # --- 1. calculate category score ---
    true_category = set(true_data.get("category", ""))
    predicted_category = set(predicted_data.get("Category", ""))
    category_score = 1.0 if true_category == predicted_category else 0.0
    
    # --- 2. calculate tissue_list score ---

    true_tissues = set(map(str.lower, true_data.get("tissue_list", [])))
    if len(true_tissues) == 0:
        true_tissues = {'low expression'}
    predicted_tissues = set(map(str.lower, predicted_data.get("Tissue", [])))

    
    if not true_tissues and not predicted_tissues:
        tissue_list_score = 1.0  
    elif len(predicted_tissues) == 0:
        tissue_list_score = 0.0
    else:
        intersection = true_tissues.intersection(predicted_tissues)
        precision = len(intersection) / len(predicted_tissues)
        recall = len(intersection) / len(true_tissues)

        if precision + recall == 0:
            tissue_list_score = 0.0
        else:
            tissue_list_score = 2 * (precision * recall) / (precision + recall)
    
    # --- 3. calculate final score ---
    final_score = (category_score * category_weight) + (tissue_list_score * tissue_list_weight)
    
    return final_score

