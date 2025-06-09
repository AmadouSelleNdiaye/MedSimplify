from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class T5_small():
    def __init__(self):
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

    def finetune(self):
        """Placeholder for finetuning functionality"""
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
