// For example, you can write variable declrations as follows:
local embedding_dim = 256;
local hidden_dim = 128;

{
  // data reader config
  "dataset_reader": {
    "type": "SemEval2017-Task4-SubsetA",
    "token_indexers": {
      "tokens": {
        "type": "elmo_characters"
      }
    }
  },
  "train_data_path": "dataset/train/",
  "validation_data_path": "dataset/test/",
  
  // model config
  "model": {
    "type": "semeval_classifier",
    "token_embedders": {
        "type": "elmo_token_embedder",
        "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json",
        "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
        "do_layer_norm": false,
        "dropout": 0.5
      }
    },
    "title_encoder": {
      "type": "lstm",
      "embedding_dim": embedding_dim,
      "hidden_size": hidden_dim
    }
  },

  // data iterator config
  "iterator": {
    "type": "bucket",
    "sorting_keys":  [["tokens", "num_tokens"]],
    "batch_size": 64
  },

  // trainer config
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": 0,
    "optimizer": {
      "type": "adam"
    }
  }
}