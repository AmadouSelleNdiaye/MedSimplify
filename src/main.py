import argparse
from model import t5_small
def argument_parser():
    """
        A parser to allow user to easily experiment different models for fine-tuning along with the med-easi dataset and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 main.py [model] [mode] [dataset] [quantization]'
                                           '\n python3 main.py --model t5_small train --dataset med_easi',
                                     description="This program allows to fine-tune different models on Med-Easi dataset.",
                                     add_help=True)
    
    parser.add_argument('--model', type=str, default="t5_small",
                        choices=["t5_small","bart_base", "distillBART", "scifive_base", "bart_large_cnn"],
                        help="Name of the file containing model used to finetune or to make "
                             " prediction on test data")
    
    parser.add_argument('--mode', type=str, default="finetune",choices=["finetune","predict"],
                        help="Mode for model operation: 'finetune' for finetuning the model, " \
                        "'predict' for generating predictions on new data")
    
    parser.add_argument('--dataset', type=str, default="med_easi")
   
 
    
    parser.add_argument('--quantization', type=str, choices=["qa_4b","qa_8b"],
                        help="Quantization precision: 'qa_4b' for 4-bit quantization, " \
                        "'qa_8b' for 8-bit quantization")
    return parser.parse_args()

if __name__=="__main__":
    args = argument_parser()

    model = args.model
    mode = args.mode
    dataset = args.dataset
    quantization = args.quantization