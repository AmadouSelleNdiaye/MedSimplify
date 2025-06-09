from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class T5_small():
    def __init__(self,model:str,tokenizer:str):
        """
        Initialize the T5_small model with the given model and tokenizer.
        
        Parameters
        ----------
        model : str
            The name of the model to use.
        tokenizer : str
            The name of the tokenizer to use.
        """
         
        self.model= AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    def finetune():
        pass

    def predict():
        pass