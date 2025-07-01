# Fine-tuning T5 avec LoRA pour la simplification de textes médicaux
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch

# Configuration LoRA
def get_lora_config():
    """Configuration LoRA optimisée pour T5"""
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,  # Type de tâche pour T5
        inference_mode=False,             # Mode d'entraînement
        r=16,                            # Rang de la décomposition (plus élevé = plus de paramètres)
        lora_alpha=32,                   # Facteur de mise à l'échelle LoRA
        lora_dropout=0.1,                # Dropout pour LoRA
        # Modules cibles pour T5 (encoder et decoder)
        target_modules=[
            "q", "v", "k", "o",          # Attention layers
            "wi_0", "wi_1", "wo"         # Feed-forward layers
        ],
        bias="none",                     # Ne pas adapter les biais
    )

def prepare_data():
    """Préparation et tokenisation des données"""
    print("Chargement et préparation des données...")
    dataset = load_dataset("cbasu/Med-EASi")
    
    prefix = "simplify: "
    
    def preprocess(example):
        return {
            "input_text": prefix + example["Expert"],
            "target_text": example["Simple"]
        }
    
    dataset = dataset.map(preprocess)
    train_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_data = train_dataset["train"]
    eval_data = train_dataset["test"]
    
    return train_data, eval_data

def tokenize_data(train_data, eval_data, tokenizer):
    """Tokenisation des données"""
    print("Tokenisation des données...")
    
    MAX_INPUT = 512
    MAX_TARGET = 128
    
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input_text"], 
            max_length=MAX_INPUT, 
            padding="max_length", 
            truncation=True
        )
        
        labels = tokenizer(
            examples["target_text"], 
            max_length=MAX_TARGET, 
            padding="max_length", 
            truncation=True
        )
        
        # Remplacer les tokens de padding par -100 pour ignorer dans la loss
        labels_input_ids = labels["input_ids"]
        labels_input_ids = [
            [(token if token != tokenizer.pad_token_id else -100) for token in label]
            for label in labels_input_ids
        ]
        model_inputs["labels"] = labels_input_ids
        return model_inputs
    
    # Tokenisation avec suppression des colonnes originales
    train_tokenized = train_data.map(
        tokenize_function, 
        batched=True,
        remove_columns=train_data.column_names
    )
    
    eval_tokenized = eval_data.map(
        tokenize_function, 
        batched=True,
        remove_columns=eval_data.column_names
    )
    
    print(f"Colonnes après tokenisation: {train_tokenized.column_names}")
    return train_tokenized, eval_tokenized

def setup_lora_model(model_name="t5-small"):
    """Configuration du modèle avec LoRA"""
    print("Configuration du modèle avec LoRA...")
    
    # Charger le modèle de base
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Appliquer LoRA
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    
    # Afficher les paramètres entraînables
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_lora_model():
    """Fonction principale d'entraînement avec LoRA"""
    
    # 1. Préparation des données
    train_data, eval_data = prepare_data()
    
    # 2. Configuration du modèle LoRA
    model, tokenizer = setup_lora_model("t5-small")
    
    # 3. Tokenisation
    train_tokenized, eval_tokenized = tokenize_data(train_data, eval_data, tokenizer)
    
    # 4. Configuration de l'entraînement
    training_args = Seq2SeqTrainingArguments(
        output_dir="./t5-med-simplify-lora",
        save_strategy="epoch",
        eval_strategy="epoch",
        per_device_train_batch_size=8,        # Batch size plus conservateur
        per_device_eval_batch_size=8,
        num_train_epochs=10,                  # Moins d'époques nécessaires avec LoRA
        learning_rate=1e-3,                   # Learning rate plus élevé pour LoRA
        weight_decay=0.01,
        warmup_steps=100,
        predict_with_generate=True,
        logging_dir="./logs-lora",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,                           # Utiliser la précision mixte
        report_to="none",
        gradient_accumulation_steps=2,        # Compenser la réduction du batch size
        dataloader_drop_last=True,
        remove_unused_columns=False,          # Important pour LoRA
        dataloader_num_workers=0,            # Éviter les problèmes de multiprocessing
        include_num_input_tokens_seen=False, # Correction pour la compatibilité
    )
    
    # 5. Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # 6. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 7. Entraînement
    print("Début de l'entraînement avec LoRA...")
    trainer.train()
    
    # 8. Sauvegarde du modèle LoRA
    print("Sauvegarde du modèle LoRA...")
    model.save_pretrained("./t5-med-simplify-lora")
    tokenizer.save_pretrained("./t5-med-simplify-lora")
    
    print("Entraînement LoRA terminé !")
    return model, tokenizer

def load_lora_model(model_path="./t5-med-simplify-lora", base_model="t5-small"):
    """Charger un modèle LoRA sauvegardé"""
    print("Chargement du modèle LoRA...")
    
    # Charger le modèle de base
    base_model = T5ForConditionalGeneration.from_pretrained(base_model)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    
    # Charger les adaptateurs LoRA
    model = PeftModel.from_pretrained(base_model, model_path)
    
    print("Modèle LoRA chargé avec succès!")
    return model, tokenizer

def test_lora_model(model, tokenizer, text, prefix="simplify: "):
    """Tester le modèle LoRA - VERSION CORRIGÉE"""
    try:
        device = next(model.parameters()).device
        
        input_text = prefix + text
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,  # CORRECTION: argument nommé au lieu de positionnel
                attention_mask=attention_mask,  # Ajout du masque d'attention
                max_length=128,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    except Exception as e:
        print(f"Erreur lors de la génération LoRA: {e}")
        return f"Erreur: {str(e)}"

def test_base_model(model, tokenizer, text, prefix="simplify: "):
    """Fonction spécifique pour tester le modèle de base T5"""
    try:
        device = next(model.parameters()).device
        
        input_text = prefix + text
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,  # Le modèle T5 de base accepte les arguments positionnels
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    except Exception as e:
        print(f"Erreur modèle base: {e}")
        return f"Erreur: {str(e)}"

def compare_models():
    """Comparer le modèle LoRA avec le modèle de base - VERSION CORRIGÉE"""
    print("=== Comparaison des modèles ===")
    
    # Charger le modèle LoRA
    try:
        lora_model, lora_tokenizer = load_lora_model()
        print("✓ Modèle LoRA chargé")
    except Exception as e:
        print(f"✗ Impossible de charger le modèle LoRA: {e}")
        return
    
    # Charger le modèle de base pour comparaison
    try:
        base_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        base_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        print("✓ Modèle de base chargé")
    except Exception as e:
        print(f"✗ Erreur chargement modèle base: {e}")
        return
    
    # Textes de test
    test_texts = [
        "The patient presents with acute myocardial infarction secondary to coronary artery occlusion.",
        "Bilateral pulmonary infiltrates consistent with pneumonia were observed on chest radiography.",
        "CT also is required to accurately assess skull base bony changes, which are less visible on MRI."
    ]
    
    print("\n=== Résultats de comparaison ===")
    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i} ---")
        print(f"Original: {text}")
        
        # Test modèle LoRA avec la fonction corrigée
        lora_result = test_lora_model(lora_model, lora_tokenizer, text)
        print(f"LoRA: {lora_result}")
        
        # Test modèle de base
        base_result = test_base_model(base_model, base_tokenizer, text)
        print(f"Base: {base_result}")

# Fonctions utilitaires pour l'analyse
def analyze_lora_model(model_path="./t5-med-simplify-lora"):
    """Analyser les paramètres du modèle LoRA"""
    try:
        from peft import PeftModel
        base_model = T5ForConditionalGeneration.from_pretrained("t5-small")
        lora_model = PeftModel.from_pretrained(base_model, model_path)
        
        print("=== Analyse du modèle LoRA ===")
        lora_model.print_trainable_parameters()
        
        # Informations sur les adaptateurs
        print(f"\nAdaptateurs actifs: {lora_model.active_adapters}")
        print(f"Configuration LoRA: {lora_model.peft_config}")
        
    except Exception as e:
        print(f"Erreur lors de l'analyse: {e}")

def quick_test():
    """Test rapide sans entraînement complet"""
    print("=== Test rapide du modèle LoRA ===")
    
    try:
        # Essayer de charger un modèle existant
        model, tokenizer = load_lora_model()
        print("Modèle LoRA existant chargé.")
    except:
        print("Aucun modèle LoRA trouvé. Veuillez d'abord entraîner le modèle.")
        return
    
    # Test simple
    test_text = "The patient presents with acute myocardial infarction."
    result = test_lora_model(model, tokenizer, test_text)
    print(f"\nTexte original: {test_text}")
    print(f"Texte simplifié: {result}")

def interactive_test():
    """Mode interactif pour tester le modèle"""
    print("=== Mode interactif ===")
    print("Tapez 'quit' pour sortir")
    
    try:
        model, tokenizer = load_lora_model()
        print("✓ Modèle LoRA chargé")
    except:
        print("✗ Impossible de charger le modèle LoRA")
        return
    
    while True:
        text = input("\nEntrez un texte médical à simplifier: ")
        if text.lower() == 'quit':
            break
        
        if text.strip():
            result = test_lora_model(model, tokenizer, text)
            print(f"Résultat: {result}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "train":
            # Entraîner le modèle LoRA
            print("Démarrage de l'entraînement LoRA...")
            model, tokenizer = train_lora_model()
            
            # Tester le modèle
            print("\nTest du modèle LoRA...")
            test_text = "The patient presents with acute myocardial infarction."
            result = test_lora_model(model, tokenizer, test_text)
            print(f"Résultat: {result}")
            
        elif mode == "test":
            quick_test()
            
        elif mode == "compare":
            compare_models()
            
        elif mode == "analyze":
            analyze_lora_model()
            
        elif mode == "interactive":
            interactive_test()
            
        else:
            print("Modes disponibles: train, test, compare, analyze, interactive")
    
    else:
        # Mode par défaut: entraînement complet
        print("Démarrage de l'entraînement LoRA...")
        model, tokenizer = train_lora_model()
        
        # Tester le modèle
        print("\nTest du modèle LoRA...")
        test_text = "The patient presents with acute myocardial infarction."
        result = test_lora_model(model, tokenizer, test_text)
        print(f"Résultat: {result}")
        
        # Comparer avec le modèle de base
        print("\nComparaison des modèles...")
        compare_models()
        
        # Analyser le modèle
        print("\nAnalyse du modèle...")
        analyze_lora_model()