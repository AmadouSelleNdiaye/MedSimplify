from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



class Bart_large_cnn():
    def __init__(self,model:str,tokenizer:str):
        """
        Initialize the Bart_large_cnn model with the given model and tokenizer.
        
        Parameters
        ----------
        model : str
            The name of the model to use.
        tokenizer : str
            The name of the tokenizer to use.
        """
        self.model= AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        
    def finetune():
        pass

    def predict():
        pass