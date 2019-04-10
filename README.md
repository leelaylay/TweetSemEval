
# SemEval-2017 Task 4 Sentiment Analysis in Twitter

## Introduction
[SemEval-2017 Task 4](http://alt.qcri.org/semeval2017/task4/index.php?id=data-and-tools)

## Run Experiments

```bash
conda create -n allennlp python=3.6
source activate allennlp
pip install -r requirements.txt
```

```bash
chmod +x run.sh  && ./run.sh
```

## Experiments Results
| Models                                      | F1-score |
|---------------------------------------------|----------|
| LSTMs+CNNs ensemble with multiple conv. ops | 0.685    |
| Deep Bi-LSTM+attention                      | 0.677    |
| Our Model |    |
