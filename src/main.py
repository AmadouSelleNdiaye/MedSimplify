import argparse

def argument_parser():
    """
        A parser to allow user to easily experiment different models for fine-tuning along with the med-easi dataset and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 main.py [model] [dataset] [hyper_parameters]'
                                           '\n python3 main.py --model t5_small --predict',
                                     description="This program allows to fine-tune different models on Med-Easi dataset.",
                                     add_help=True)
    parser.add_argument('--model', type=str, default="t5_small",
                        choices=["t5_small","bart_base", "distillBART", "scifive_base", "bart_large_cnn"])
    parser.add_argument('--dataset', type=str, default="med_easi")
    parser.add_argument('--quantization', type=str, default="qa_4b", choices=["qa_4b","qa_8b"])
 
    parser.add_argument('--predict', type=str,
                        help="Name of the file containing model weights used to make "
                             " prediction on test data")
    return parser.parse_args()

if __name__=="__main__":
    pass