import math, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils import device, timer

model_name = "gpt2-medium"  # 355M baseline
tok = AutoTokenizer.from_pretrained(model_name)
tok.pad_token = tok.eos_token

with timer("load_model"):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device())
    model.eval()

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts = "\n\n".join(ds["text"])
enc = tok(texts, return_tensors="pt")
input_ids = enc["input_ids"].to(device())

with torch.no_grad(), timer("forward_ppl"):
    # Sliding window to avoid OOM
    stride, nlls, max_len = 1024, [], input_ids.size(1)
    for i in range(0, max_len, stride):
        end = min(i+stride, max_len)
        trg_len = end - i
        inp_ids = input_ids[:, i:end]
        tgt_ids = inp_ids.clone()
        out = model(inp_ids, labels=tgt_ids)
        nlls.append(out.loss * trg_len)
    ppl = math.exp(torch.stack(nlls).sum()/max_len)
print({"perplexity": round(float(ppl), 2)})
