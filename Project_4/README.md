# Project 4 Image Captioning Using Traditional LSTM & RNN Methods
## Learning Outcome
This project taught me how to utilize RNN and LSTM architectures to generate captions for images. This was my first NLP project.

## Code Description
* Define the configuration for your experiment. See `default.json` to see the structure and available options. 
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir. This can be configured in `contants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training 
pr evaluate performance.
* If you want to test any model, uncomment the testing line in main.py, be sure to give model_path as an argument to exp.test()
* If you want to train the model with data augmentation, be sure to call CocoDataset with transform = True for training set.

Files
-----
- main.py: Main driver class
- model.py: Model architecture class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging 
and resuming experiments.
- dataset_factory: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 
- caption_utils.py: utility functions to generate bleu scores
- vocab.py: A simple Vocabulary wrapper
- coco_dataset: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- get_datasets.ipynb: A helper notebook to set up the dataset in your workspace