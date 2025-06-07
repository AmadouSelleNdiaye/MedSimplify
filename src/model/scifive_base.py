from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-base-Pubmed")
model = AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-base-Pubmed")