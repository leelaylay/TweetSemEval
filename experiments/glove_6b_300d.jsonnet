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
    "type": "semeval_classifier",
    "word_embeddings": {
        "tokens": {
          "type": "embedding",
          "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.300d.txt.gz",
          "embedding_dim": embedding_dim,
          "trainable": false
      },
      
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
    "batch_size": 512
  },

  // trainer config
  "trainer": {
    "num_epochs": 40,
    "patience": 10,
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
    }
  }
}