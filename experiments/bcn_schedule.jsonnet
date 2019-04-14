// For example, you can write variable declrations as follows:
local embedding_dim = 100;
local hidden_dim = 200;

{
  // data reader config
  "dataset_reader": {
    "type": "SemEval2017-Task4-SubsetA",
    "token_indexers": {
      "tokens": {
        "type": "single_id"
      },
      "elmo": {
        "type": "elmo_characters"
      }
    }
  },
  "train_data_path": "dataset/train/",
  "validation_data_path": "dataset/test/",
  
  // model config
  "model": {
    "type": "my_bcn",
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
    "embedding_dropout": 0.5,
    "pre_encode_feedforward": {
        "input_dim": 1324,
        "num_layers": 1,
        "hidden_dims": [300],
        "activations": ["relu"],
        "dropout": [0.25]
    },
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
    "elmo": {
      "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
      "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
      "do_layer_norm": false,
      "dropout": 0.0,
      "num_output_representations": 1
    },
    "use_input_elmo": true,
    "use_integrator_output_elmo": false,
    "output_layer": {
        "input_dim": 2400,
        "num_layers": 3,
        "output_dims": [1200, 600, 3],
        "pool_sizes": 4,
        "dropout": [0.2, 0.3, 0.0]
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
    "num_epochs": 150,
    "cuda_device": 0,
    "grad_norm": 5.0,
    "learning_rate_scheduler": {
          "type": "cosine",
          "t_initial": 500,
          "t_mul": 1.5,
          "eta_min": 0.0,
          "eta_mul": 0.7
      },
    "optimizer": {
      "type": "adam",
      "lr": 0.0001
    }
  }
}