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