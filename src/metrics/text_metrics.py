import numpy as np
import re

class Textmetrics():
    def __init__(self):
        pass

    def tokenize(self,text: str)-> list[str]:
        #Processing of text
        text = text.lower().strip()
        #Tokenization of the text
        tokens =re.findall(r'b\w+\b',text)

        return tokens
    
    def n_grams(self, tokens: list[str], n:int)-> list[str,...]:
        """Generate of n_grams from the list of tokens"""
        if len(tokens)<n:
            return []
        
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) -n+1)]
        
    