import torch

def simplify_text(text, model, tokenizer, max_length=128):
    """
    Simplify medical text using the trained model
    """
    # Add the same prefix used during training
    input_text = "simplify: " + text
    
    # Tokenize
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True, 
        padding=True
    )
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            do_sample=False
        )
    
    # Decode
    simplified_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return simplified_text