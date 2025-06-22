import numpy as np
from datasets import load_dataset
from collections import Counter
import numpy as np


# BLUE (n-gram 1 to 4, geometric mean + brevity penalty)
def n_gram_precision(pred, ref, n):
    pred_ngrams = Counter([tuple(pred[i:i+n]) for i in range(len(pred)-n+1)])
    ref_ngrams = Counter([tuple(ref[i:i+n]) for i in range(len(ref)-n+1)])
    overlap = sum((pred_ngrams & ref_ngrams).values())
    total = max(sum(pred_ngrams.values()), 1)
    return overlap / total

def compute_bleu(pred, ref):
    pred = pred.split()
    ref = ref.split()
    precisions = [n_gram_precision(pred, ref, n) for n in range(1, 5)]
    if all(p == 0 for p in precisions):
        bleu = 0
    else:
        score = np.exp(np.mean([np.log(p + 1e-9) for p in precisions]))
        bp = np.exp(1 - len(ref)/len(pred)) if len(pred) < len(ref) else 1
        bleu = bp * score
    return bleu

# ROUGE-L (LCS-basiert F1)
def lcs(X, Y):
    m, n = len(X), len(Y)
    L = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                L[i+1][j+1] = L[i][j] + 1
            else:
                L[i+1][j+1] = max(L[i+1][j], L[i][j+1])
    return L[m][n]

def compute_rouge_l(pred, ref):
    pred_tokens, ref_tokens = pred.split(), ref.split()
    lcs_len = lcs(pred_tokens, ref_tokens)
    prec = lcs_len / max(len(pred_tokens), 1)
    rec = lcs_len / max(len(ref_tokens), 1)
    if prec + rec == 0:
        return 0
    return 2 * prec * rec / (prec + rec)

# SARI (Add, Keep, Delete F1 over n-grams)
def ngrams(s, n):
    return set([' '.join(s[i:i+n]) for i in range(len(s)-n+1)])

def compute_sari(source, pred, ref):
    source_tokens = source.split()
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    score_add, score_keep, score_del = [], [], []

    for n in range(1, 5):
        S = ngrams(source_tokens, n)
        P = ngrams(pred_tokens, n) 
        R = ngrams(ref_tokens, n)

        add = P - S
        keep = P & S
        del_ = S - P

        add_prec = len(add & R) / max(len(add), 1)
        add_rec = len(add & R) / max(len(R - S), 1)
        f1_add = 2 * add_prec * add_rec / (add_prec + add_rec + 1e-9)

        keep_prec = len(keep & R) / max(len(keep), 1)
        keep_rec = len(keep & R) / max(len(S & R), 1)
        f1_keep = 2 * keep_prec * keep_rec / (keep_prec + keep_rec + 1e-9)

        del_prec = len(del_ - R) / max(len(del_), 1)
        del_rec = len(del_ - R) / max(len(S - R), 1)
        f1_del = 2 * del_prec * del_rec / (del_prec + del_rec + 1e-9)

        score_add.append(f1_add)
        score_keep.append(f1_keep)
        score_del.append(f1_del)

    return (np.mean(score_add) + np.mean(score_keep) + np.mean(score_del)) / 3


# Load the dataset
dataset = load_dataset("cbasu/Med-EASi", split="train")
sources = dataset["Expert"]
references = dataset["Simple"]

# Calculating scores for the whole set
bleu_scores = []
rouge_scores = []
sari_scores = []

for src, ref in zip(sources, references):
    bleu_scores.append(compute_bleu(ref, src))
    rouge_scores.append(compute_rouge_l(ref, src))
    sari_scores.append(compute_sari(src, ref, ref))

print("BLEU moyenne:", np.mean(bleu_scores))
print("ROUGE-L moyenne:", np.mean(rouge_scores))
print("SARI moyenne :", np.mean(sari_scores))