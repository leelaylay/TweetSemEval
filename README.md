
# SemEval-2017 Task 4 Sentiment Analysis in Twitter

## Introduction
[SemEval-2017 Task 4](http://alt.qcri.org/semeval2017/task4/index.php?id=data-and-tools)

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


| Models                                      | F1-score |
|---------------------------------------------|----------|
| GloVe(No Grad 6B100D) + 2 BiLSTM  |  0.548   |
| GloVe(No Grad 6B300D) + 2 BiLSTM  |  0.547   |
| GloVe(No Grad 42B300D) + 2 BiLSTM  |  0.559   |
| GloVe(No Grad 840B300D) + 2 BiLSTM  |  0.629   |
| GloVe(No Grad 27B200D Twitter) + 2 BiLSTM  |  0.586   |


| Models                                      | F1-score |
|---------------------------------------------|----------|
| LSTMs+CNNs ensemble with multiple conv. ops (**SOTA**) | **0.685**    |
| Deep Bi-LSTM+attention                      | 0.677    |
| GloVe(No Grad) + 2 BiLSTM  |  0.629   |
| GloVe(With Grad) + 2 BiLSTM  |  0.642   |
| ELMo(No Grad) + 2 BiLSTM  |  0.614   |
| ELMo(With Grad) + 2 BiLSTM  |  0.609   |
| BERT(No Grad) + 2 BiLSTM  | 0.599   |
| BERT(With Grad) + 2 BiLSTM  |  0.480  |
| GloVe+ Bi-LSTM+attention  |  0.665   |
| GloVe+ ELMo + BCN  |  0.682   |
