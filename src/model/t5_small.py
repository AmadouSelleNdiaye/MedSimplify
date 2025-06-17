from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,BitsAndBytesConfig
import torch


class T5_small():
    def __init__(self,quantization:str=None):
        """
        Initialize the T5_small model with the given model and tokenizer.
        
        Parameters
        ----------
        model : str
            The name of the model to use.
        tokenizer : str
            The name of the tokenizer to use.
        quantization : str
            Quantization type: 'qa_4b' for 4-bit, 'qa_8b' for 8-bit, None for no quantization
        """
        quantization_config = None
        if quantization == "qa_4b":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif quantization == "qa_8b":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        
        # Load model with or without quantization
        if quantization_config:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-small",
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
        
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
