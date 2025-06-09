from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



class Bart_base():
    def __init__(self,model:str,tokenizer:str):
        """
        Initialize the Bart_base model with the given model and tokenizer.
        
        Parameters
        ----------
        model : str
            The name of the model to use.
        tokenizer : str
            The name of the tokenizer to use.
        """
        self.model= AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        
    def finetune(self):
        pass

    def predict(self,input_text:str):
        """
        Generate predictions for the given input text.
        
        Parameters
        ----------
        input_text : str
            The input text to generate predictions for
            
        Returns
        -------
        str
            The generated prediction
        """
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)