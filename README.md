
# SemEval-2017 Task 4 Sentiment Analysis in Twitter

## Introduction
[SemEval-2017 Task 4](http://alt.qcri.org/semeval2017/task4/index.php?id=data-and-tools) is a text sentiment classification task: Given a message, classify whether the message is of positive, negative, or neutral sentiment.


## Run Experiments

```bash
# install the environment
conda create -n allennlp python=3.6
source activate allennlp
pip install -r requirements.txt
```

```bash
# run the program
chmod +x run.sh  && ./run.sh
```

## Experiments Results
In the table, the first two models are the SOTA models for this task (ensemble and single)(as far as I know). I use AllenNLP to implement the other models below.

I found that GloVe is pretty for single embedding compared to the other embeddings. The model `GloVe + Bi-LSTM + attention` is similar to the model `Deep Bi-LSTM+attention`, and the results are similar. The model `GloVe + ELMo + BCN` achieves 0.682 F1 score with single model which is also compatible to the model `LSTMs+CNNs ensemble with multiple conv. ops`.

| Models                                      | F1-score |
|---------------------------------------------|----------|
| [LSTMs+CNNs ensemble with multiple conv. ops](https://arxiv.org/abs/1704.06125) (**Ensemble**) | **0.685**    |
| [Deep Bi-LSTM+attention](https://www.aclweb.org/anthology/papers/S/S17/S17-2126/)    (**Single model**)     | 0.677    |
| GloVe(No Grad) + 2 BiLSTM  |  0.629   |
| GloVe(With Grad) + 2 BiLSTM  |  0.642   |
| ELMo(No Grad) + 2 BiLSTM  |  0.614   |
| ELMo(With Grad) + 2 BiLSTM  |  0.609   |
| BERT(No Grad) + 2 BiLSTM  | 0.599   |
| BERT(With Grad) + 2 BiLSTM  |  0.480  |
| GloVe + Bi-LSTM + attention  |  0.665   |
| GloVe + ELMo + BCN  |  0.682   |


## Reference
1. Rosenthal S, Farra N, Nakov P. SemEval-2017 task 4: Sentiment analysis in Twitter[C]//Proceedings of the 11th international workshop on semantic evaluation (SemEval-2017). 2017: 502-518.
2. Baziotis C, Pelekis N, Doulkeridis C. Datastories at semeval-2017 task 4: Deep lstm with attention for message-level and topic-based sentiment analysis[C]//Proceedings of the 11th international workshop on semantic evaluation (SemEval-2017). 2017: 747-754.
3. Cliche M. BB_twtr at SemEval-2017 task 4: twitter sentiment analysis with CNNs and LSTMs[J]. arXiv preprint arXiv:1704.06125, 2017.
