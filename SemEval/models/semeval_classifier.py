from typing import Dict, Optional

import numpy
import torch
import torch.nn.functional as F
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides


@Model.register("semeval_classifier")
class SemEvalClassifier(Model):
    """This ``Model`` performs text classification for SemEval 2017 task 4 subset A.
    """

    def __init__(self, word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder, vocab: Vocabulary) -> None:
        super().__init__(vocab)
        # We need the embeddings to convert word IDs to their vector representations
        self.word_embeddings = word_embeddings

        self.encoder = encoder

        # After converting a sequence of vectors to a single vector, we feed it into
        # a fully-connected linear layer to reduce the dimension to the total number of labels.
        self.linear = torch.nn.Linear(
            in_features=encoder.get_output_dim(),
            out_features=vocab.get_vocab_size('labels'))

        # Monitor the metrics - we use accuracy, as well as prec, rec, f1 for 4 (very positive)
        self.accuracy = CategoricalAccuracy()
        self.f1_measure_positive = F1Measure(
            vocab.get_token_index("positive", "labels"))
        self.f1_measure_negative = F1Measure(
            vocab.get_token_index("negative", "labels"))
        self.f1_measure_neutral = F1Measure(
            vocab.get_token_index("neutral", "labels"))

        # We use the cross entropy loss because this is a classification task.
        # Note that PyTorch's CrossEntropyLoss combines softmax and log likelihood loss,
        # which makes it unnecessary to add a separate softmax layer.
        self.loss_function = torch.nn.CrossEntropyLoss()

    @overrides
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:
        # In deep NLP, when sequences of tensors in different lengths are batched together,
        # shorter sequences get padded with zeros to make them equal length.
        # Masking is the process to ignore extra zeros added by padding
        mask = get_text_field_mask(tokens)

        # Forward pass
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)

        # In AllenNLP, the output of forward() is a dictionary.
        # Your output dictionary must contain a "loss" key for your model to be trained.
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            self.f1_measure_positive(logits, label)
            self.f1_measure_negative(logits, label)
            self.f1_measure_neutral(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        _, recall_positive, f1_measure_positive = self.f1_measure_positive.get_metric(
            reset)
        _, recall_negative, f1_measure_negative = self.f1_measure_negative.get_metric(
            reset)
        _, recall_neutral, _ = self.f1_measure_neutral.get_metric(reset)
        accuracy = self.accuracy.get_metric(reset)

        avg_recall = (recall_positive + recall_negative + recall_neutral) / 3.0
        macro_avg_f1_measure = (
            f1_measure_positive + f1_measure_negative) / 2.0

        results = {
            'accuracy': accuracy,
            'avg_recall': avg_recall,
            'f1_measure': macro_avg_f1_measure
        }
        return results
