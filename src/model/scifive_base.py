from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



class Scifive_base():
    def __init__(self,model:str,tokenizer:str):
        """
        Initialize the Scifive_base model with the given model and tokenizer.
        
        Parameters
        ----------
        model : str
            The name of the model to use.
        tokenizer : str
            The name of the tokenizer to use.
        """
        self.model= AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-base-Pubmed")
        self.tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-base-Pubmed")
        
    def finetune():
        pass

    def predict():
        pass