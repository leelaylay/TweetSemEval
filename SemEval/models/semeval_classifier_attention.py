#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Dict, Optional, Union

import numpy
import torch
import torch.nn.functional as F
from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Maxout, Seq2SeqEncoder, TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides


@Model.register("semeval_classifier_attention")
class SemEvalClassifierAttention(Model):
    """This ``Model`` performs text classification for SemEval 2017 task 4 subset A.
    """

    def __init__(self,  
                vocab: Vocabulary, 
                text_field_embedder: TextFieldEmbedder,
                embedding_dropout: float,
                encoder: Seq2SeqEncoder,
                integrator: Seq2SeqEncoder,
                integrator_dropout: float,
                output_layer: Union[FeedForward, Maxout],
                initializer: InitializerApplicator = InitializerApplicator(),
                regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        # We need the embeddings to convert word IDs to their vector representations
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout)
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder

        self.integrator = integrator
        self.integrator_dropout = torch.nn.Dropout(integrator_dropout)

        self._self_attentive_pooling_projection = torch.nn.Linear(
                self.integrator.get_output_dim(), 1)
        self.output_layer = output_layer

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
        initializer(self)

    @overrides
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:
        # In deep NLP, when sequences of tensors in different lengths are batched together,
        # shorter sequences get padded with zeros to make them equal length.
        # Masking is the process to ignore extra zeros added by padding
        text_mask = util.get_text_field_mask(tokens).float()
        # Forward pass
        embedded_text = self.text_field_embedder(tokens)
        dropped_embedded_text = self.embedding_dropout(embedded_text)
        encoded_tokens = self.encoder(dropped_embedded_text, text_mask)

        # Compute biattention. This is a special case since the inputs are the same.
        attention_logits = encoded_tokens.bmm(encoded_tokens.permute(0, 2, 1).contiguous())
        attention_weights = util.masked_softmax(attention_logits, text_mask)
        encoded_text = util.weighted_sum(encoded_tokens, attention_weights)

        # Build the input to the integrator
        integrator_input = torch.cat([encoded_tokens,
                                      encoded_tokens - encoded_text,
                                      encoded_tokens * encoded_text], 2)
        integrated_encodings = self.integrator(integrator_input, text_mask)

        # Simple Pooling layers
        max_masked_integrated_encodings = util.replace_masked_values(
                integrated_encodings, text_mask.unsqueeze(2), -1e7)
        max_pool = torch.max(max_masked_integrated_encodings, 1)[0]
        min_masked_integrated_encodings = util.replace_masked_values(
                integrated_encodings, text_mask.unsqueeze(2), +1e7)
        min_pool = torch.min(min_masked_integrated_encodings, 1)[0]
        mean_pool = torch.sum(integrated_encodings, 1) / torch.sum(text_mask, 1, keepdim=True)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self._self_attentive_pooling_projection(
                integrated_encodings).squeeze(2)
        self_weights = util.masked_softmax(self_attentive_logits, text_mask)
        self_attentive_pool = util.weighted_sum(integrated_encodings, self_weights)

        pooled_representations = torch.cat([max_pool, min_pool, mean_pool, self_attentive_pool], 1)
        pooled_representations_dropped = self.integrator_dropout(pooled_representations)
        
        logits = self.output_layer(pooled_representations_dropped)

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

   
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'SemEvalClassifierAttention':  # type: ignore
        # pylint: disable=arguments-differ
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab=vocab, params=embedder_params)
        embedding_dropout = params.pop("embedding_dropout")
        encoder = Seq2SeqEncoder.from_params(params.pop("encoder"))
        integrator = Seq2SeqEncoder.from_params(params.pop("integrator"))
        integrator_dropout = params.pop("integrator_dropout")

        output_layer_params = params.pop("output_layer")
        if "activations" in output_layer_params:
            output_layer = FeedForward.from_params(output_layer_params)
        else:
            output_layer = Maxout.from_params(output_layer_params)

        elmo = params.pop("elmo", None)
        if elmo is not None:
            elmo = Elmo.from_params(elmo)
        use_input_elmo = params.pop_bool("use_input_elmo", False)
        use_integrator_output_elmo = params.pop_bool("use_integrator_output_elmo", False)

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))
        params.assert_empty(cls.__name__)

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   embedding_dropout=embedding_dropout,
                   encoder=encoder,
                   integrator=integrator,
                   integrator_dropout=integrator_dropout,
                   output_layer=output_layer,
                   initializer=initializer,
                   regularizer=regularizer)