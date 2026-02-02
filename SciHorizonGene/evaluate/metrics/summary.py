from transformers import AutoModelForCausalLM, AutoTokenizer
from bert_score import score as bertscore
from rouge_score import rouge_scorer
import torch
from ast import literal_eval

def calculate_summary(answer: str, prediction: str):
    
    prediction = prediction.replace('```json', '').replace('```', '').strip()
    prediction = literal_eval(prediction).get("summary", "")
    
    bert_model = "bert-base-uncased"
    PPL_MODEL_NAME = "openai-community/gpt2"  
    ppl_tokenizer = AutoTokenizer.from_pretrained(PPL_MODEL_NAME, cache_dir="./gpt-2-model/")
    ppl_model = AutoModelForCausalLM.from_pretrained(PPL_MODEL_NAME, cache_dir="./gpt-2-model/").to("cuda")
    ppl_model.eval()
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    assert ppl_model is not None and ppl_tokenizer is not None, "need to load perplexity model and tokenizer!"
    assert rouge_scorer is not None, "need to initialize rouge scorer!"
    
    # ---- ROUGE-L ----
    rouge_score = rouge.score(answer, prediction)["rougeL"].fmeasure

    # ---- BERTScore ----
    P, R, F1 = bertscore([prediction], [answer], model_type=bert_model, lang="en", verbose=False)
    bert_f1 = float(F1.mean())

    # ---- Perplexity ----
    try:
        encodings = ppl_tokenizer(prediction, return_tensors="pt").to("cuda")
        with torch.no_grad():
            max_length = ppl_model.config.n_positions
            stride = 512
            nlls = []
            for i in range(0, encodings.input_ids.size(1), stride):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = i + stride
                trg_len = end_loc - i
                input_ids = encodings.input_ids[:, begin_loc:end_loc]
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100
                outputs = ppl_model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
                nlls.append(neg_log_likelihood)
            ppl = torch.exp(torch.stack(nlls).sum() / end_loc).item()
    except Exception as e:
        print(f"Perplexity calculation error: {e}. Setting PPL to a high value.")
        print(f"The original text was: {prediction}")

        raise ValueError("Perplexity calculation failed.")

    return rouge_score, bert_f1, ppl, encodings.input_ids.size(1)

