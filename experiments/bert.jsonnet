// For example, you can write variable declrations as follows:
local embedding_dim = 768;
local hidden_dim = 100;

{
  // data reader config
  "dataset_reader": {
    "type": "SemEval2017-Task4-SubsetA",
    "token_indexers": {
      "tokens": {
        "type": "bert-pretrained",
        "pretrained_model":"bert-base-uncased",
        "do_lowercase": false,
        "use_starting_offsets": true
      }
    }
  },
  "train_data_path": "dataset/train/",
  "validation_data_path": "dataset/test/",
  
  // model config
  "model": {
    "type": "semeval_classifier",
    "word_embeddings": {
      "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {
            "tokens": ["tokens", "tokens-offsets"]
        },
        "tokens": {
          "type": "bert-pretrained",
          "pretrained_model": "bert-base-uncased",
          "requires_grad": true
        }
      },
    "encoder": {
      "type": "lstm",
      "input_size": embedding_dim,
      "hidden_size": hidden_dim,
      "num_layers": 2,
      "bidirectional": true
    }
  },

  // data iterator config
  "iterator": {
    "type": "bucket",
    "sorting_keys":  [["tokens", "num_tokens"]],
    "batch_size": 128
  },

  // trainer config
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      // "lr": 0.001
    }
  }
}