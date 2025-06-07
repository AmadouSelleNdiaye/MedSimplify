from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("zaib32/autotrain-finetuned_distillbart-3664997842")
model = AutoModelForSeq2SeqLM.from_pretrained("zaib32/autotrain-finetuned_distillbart-3664997842")