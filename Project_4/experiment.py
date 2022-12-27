################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

from cmath import inf
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import re

from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model
from vocab import *

# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__learning_rate = config_data['experiment']['learning_rate']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.
        self.__best_loss = inf

        # Init Model
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__model = get_model(config_data, self.__vocab)

        # TODO: Set these Criterion and Optimizers Correctly - DONE
        params = list(self.__model.parameters())
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.AdamW(params, lr=self.__learning_rate, weight_decay=0.0001)

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    # TODO: Perform one training iteration on the whole dataset and return loss value - DONE
    def __train(self):
        self.__model.train()
        training_loss = []

        for i, (images, captions, lengths, img_ids) in enumerate(self.__train_loader):
            self.__optimizer.zero_grad()
            images = images.to(self.__device)
            captions = captions.to(self.__device)
            targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True).data
            outputs = self.__model(images, captions, lengths)
            loss = self.__criterion(outputs, targets)
            training_loss.append(loss.item())
            loss.backward()
            self.__optimizer.step()

        return np.mean(training_loss)

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here. - DONE
    def __val(self):
        self.__model.eval()
        val_loss = 0
        loss_list = []

        with torch.no_grad():
            for i, (images, captions, lengths, img_ids) in enumerate(self.__val_loader):
                images = images.to(self.__device)
                captions = captions.to(self.__device)
                outputs = self.__model(images, captions, lengths)
                targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True).data
                loss = self.__criterion(outputs, targets)
                loss_list.append(loss.item())
            val_loss = np.mean(loss_list)
            if val_loss < self.__best_loss:
                self.__best_loss = val_loss
                self.__best_model = self.__model.state_dict()
                self.__save_model(model_path='best_model.pt')
                result_str = "Best Validation Loss: {}, Epoch: {}".format(self.__best_loss,
                                                                            self.__current_epoch)
                self.__log(result_str)

        return val_loss

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note that you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores. - DONE

    
    def test(self, model_location):
        
        self.__best_model = self.__model
        best_checkpoint = torch.load(model_location)
        self.__best_model.load_state_dict(best_checkpoint['model'])
        self.__best_model = self.__best_model.to(self.__device)
        self.__best_model.eval()
        
        test_loss = 0
        loss_list = []
        bleu1_list = []
        bleu4_list = []

        with torch.no_grad():
            for iter, (images, captions, lengths, image_ids) in enumerate(self.__test_loader):
                
                b1 = 0
                b4 = 0

                for img, ind in zip(images, image_ids):
                  references = []
                  generated_caption_orig = self.__best_model.generate_caption(img, 
                                                                          self.__vocab, 
                                                                          self.__generation_config['stochastic'],
                                                                          self.__generation_config['max_length'], 
                                                                          self.__generation_config['temperature'],
                                                                          self.__device)

                  #Generated caption to lower
                  generated_caption = [item for item in generated_caption_orig 
                                       if item not in ['<unk>', '<pad>', '<start>', '<end>', '<EOS>']]
                  
                  generated_caption = ' '.join(generated_caption).lower()

                  #Remove punctuation
                  generated_caption = re.sub(r'[^\w\s]', '', generated_caption).split()
                  
                  for annot in self.__coco_test.imgToAnns[ind]:
                      
                      #Referenced caption to lower, split
                      references.append(re.sub(r'[^\w\s]', '', annot['caption'].lower()).split())
                  
                  #Append val to overall bleu
                  b1 += bleu1(references, generated_caption)
                  b4 += bleu4(references, generated_caption)

                #Append bleu scores normed by img count
                bleu1_list.append(b1/len(images))
                bleu4_list.append(b4/len(images))
                
                #Calc loss
                images = images.to(self.__device)
                captions = captions.to(self.__device)
                outputs = self.__best_model(images, captions, lengths)
                targets = nn.utils.rnn.pack_padded_sequence(captions, lengths, batch_first=True).data
                loss = self.__criterion(outputs, targets)
                loss_list.append(loss.item())
            
            #Final scores
            test_loss = np.mean(loss_list)
            b1_final = np.mean(bleu1_list)
            b4_final = np.mean(bleu4_list)

        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss,
                                                                                b1_final,
                                                                                b4_final)
        self.__log(result_str)

        return test_loss, b1_final, b4_final

    def __save_model(self, model_path = 'latest_model.pt'):
        root_model_path = os.path.join(self.__experiment_dir, model_path)
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()