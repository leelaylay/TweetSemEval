// For example, you can write variable declrations as follows:
local embedding_dim = 300;
local hidden_dim = 100;

{
  // data reader config
  "dataset_reader": {
    "type": "SemEval2017-Task4-SubsetA"
  },
  "train_data_path": "dataset/train/",
  "validation_data_path": "dataset/test/",
  
  // model config
  "model": {
    "type": "semeval_classifier_attention",
    "text_field_embedder": {
        "token_embedders": {
          "tokens": {
              "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
              "type": "embedding",
              "embedding_dim": 300,
              "trainable": false
          }
        }
      },
      "embedding_dropout": 0.25,
      "encoder": {
        "type": "lstm",
        "input_size": 300,
        "hidden_size": 300,
        "num_layers": 1,
        "bidirectional": true
      },
      "integrator": {
        "type": "lstm",
        "input_size": 1800,
        "hidden_size": 300,
        "num_layers": 1,
        "bidirectional": true
      },
      "integrator_dropout": 0.1,
      "output_layer": {
          "input_dim": 2400,
          "num_layers": 2,
          "output_dims": [100, 3],
          "pool_sizes": 4,
          "dropout": [0.2, 0.0]
      }
  },

  // data iterator config
  "iterator": {
    "type": "bucket",
    "sorting_keys":  [["tokens", "num_tokens"]],
    "batch_size": 512
  },

  // trainer config
  "trainer": {
    "num_epochs": 50,
    "patience": 5,
    "grad_norm": 5.0,
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.0001
    }
  }
}