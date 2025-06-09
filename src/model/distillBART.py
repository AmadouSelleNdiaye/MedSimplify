from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Distillbart():
    def __init__(self,model:str,tokenizer:str):
        """
        Initialize the Distillbart model with the given model and tokenizer.
        
        Parameters
        ----------
        model : str
            The name of the model to use.
        tokenizer : str
            The name of the tokenizer to use.
        """
         
        self.model= AutoModelForSeq2SeqLM.from_pretrained("zaib32/autotrain-finetuned_distillbart-3664997842")
        self.tokenizer = AutoTokenizer.from_pretrained("zaib32/autotrain-finetuned_distillbart-3664997842")
        

    def finetune():
        pass

    def predict():
        pass