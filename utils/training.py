import time
import torch
import logging
from utils.HSCIC import HSCIC  
from tqdm import tqdm 

class TrainingBase:
    def __init__(self, model, optimizer, loss_function, num_epochs, beta_hscic=0.1, beta_l=1.0):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.num_epochs = num_epochs
        self.beta_hscic = beta_hscic
        self.beta_l = beta_l
        self.hscic = HSCIC()

    def train(self, train_dataloader, test_dataloader):
        start_time = time.time()
        total_loss = {
            "train_loss": [],
            "train_acc": [],
            "train_hscic": [],
            "test_loss": [],
            "test_acc": [],
            "test_hscic": []
        }
           
        pbar = tqdm(range(self.num_epochs), desc='Training Progress')
        for epoch in pbar:
            train_loss = self._run_epoch(train_dataloader, is_train=True)
            test_loss = self._run_epoch(test_dataloader, is_train=False)

            pbar.set_description(
                f"Epoch {epoch + 1}/{self.num_epochs} | Train Loss: {train_loss['loss']:.4f} | "
                f"Train Acc: {train_loss['acc']:.4f} | Test Loss: {test_loss['loss']:.4f} | "
                f"Test Acc: {test_loss['acc']:.4f}"
            )

            total_loss["train_loss"].append(train_loss["loss"])
            total_loss["train_acc"].append(train_loss["acc"])
            total_loss["train_hscic"].append(train_loss["hscic"])

            total_loss["test_loss"].append(test_loss["loss"])
            total_loss["test_acc"].append(test_loss["acc"])
            total_loss["test_hscic"].append(test_loss["hscic"])
                    
            if (epoch % 20 == 0):
                self._logging_stats(epoch, train_loss, test_loss)
            
        duration = time.time() - start_time
        print(f'Training completed in {duration:.2f} seconds')
        return total_loss

    def _run_epoch(self, dataloader, is_train=True):
        stats = {"loss": 0.0, "hscic": 0.0, "acc": 0.0}
        count = len(dataloader)
        for data in dataloader:
            inputs, outputs = self._prepare_data(data)
            self.optimizer.zero_grad()
  
            predictions = self.model(inputs)
            predictions = predictions.squeeze(1)

            loss = self.loss_function(predictions, outputs)
            hscic_value = self._calculate_hscic(inputs, predictions)
            combined_loss = self.beta_l * loss + self.beta_hscic * hscic_value
            
            if is_train:
                combined_loss.backward()
                self.optimizer.step()
            
            stats["loss"] += combined_loss.item()
            stats["hscic"] += hscic_value.item()
            stats["acc"] += loss.item()

        for key in stats:
            stats[key] /= count
            
        return stats

    def _calculate_hscic(self, inputs, predictions):
        pass
    
    def _prepare_data(self, data):
        pass
    
    def _logging_stats(self, epoch, train_loss, test_loss):
            logging.info(
                f'Epoch {epoch + 1}, Train Total Loss: {train_loss["loss"]:.4f}, '
                f'Train Accuracy: {train_loss["acc"]:.4f}, Train HSCIC: {train_loss["hscic"]:.4f}, '
                f'Test Total Loss: {test_loss["loss"]:.4f}, Test Accuracy: {test_loss["acc"]:.4f}, '
                f'Test HSCIC: {test_loss["hscic"]:.4f}'
            )

class StandardTraining(TrainingBase):
        def _prepare_data(self, data):
            inputs, outputs = inputs, outputs = torch.split(data[0], (data[0].shape[1]-1), 1)
            outputs = outputs.squeeze(1)
            return inputs, outputs
        
        def _calculate_hscic(self, inputs, predictions):
            return self.hscic(predictions, inputs[:,2], inputs[:, 0:1]) 

class Level2(TrainingBase):
    def _prepare_data(self, data):
        inputs = torch.split(data[0], (2, 2, 1), 1)
        return inputs, data[1]

    def _calculate_hscic(self, inputs, predictions):
        return self.hscic(predictions, inputs[2], torch.cat([inputs[0], inputs[1]], dim=1))

class ImageTraining(TrainingBase):
    def _prepare_data(self, data):
        imgs, tabular, output = data
        return (imgs, tabular), output

    def _calculate_hscic(self, inputs, predictions):
        _, tabular = inputs
        return self.hscic(predictions, tabular[:, 4], torch.cat((tabular[:, 1].unsqueeze(1), tabular[:, 5].unsqueeze(1), tabular[:, 2].unsqueeze(1)), dim=1))


