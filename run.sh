
export CUDA_VISIBLE_DEVICES=0
allennlp train experiments/default.json -s output --include-package SemEval
