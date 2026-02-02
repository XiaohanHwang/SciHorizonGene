from ast import literal_eval

def parse_prediction(prediction:str):
    prediction = prediction.strip()
    pre = prediction.replace('```json', '').replace('```', '').strip()
    return pre

def calculate_single_choice(answer, prediction):
    """
    calculate single choice
    """
    prediction = parse_prediction(prediction)
    prediction = literal_eval(prediction)
    prediction = prediction.get('answer', '').strip()
    
    return 1.0 if answer == prediction else 0.0

def calculate_multiple_choice(answer, prediction):
    """
    calculate multiple choice
    """
    prediction = parse_prediction(prediction)
    prediction = literal_eval(prediction)
    prediction = prediction.get('answers', '')
    
    if answer and prediction:
        tp = len(set(answer) & set(prediction))
        fp = len(set(prediction) - set(answer))
        fn = len(set(answer) - set(prediction))
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    else:
        f1 = 0.0
        
    return f1
    
