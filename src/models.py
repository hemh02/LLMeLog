import os
import torch
import numpy as np
import torch.nn as nn
import logging
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from src.transformerEncoder import make_model
from transformers import AutoModel, AutoTokenizer
from src.CosFace import MarginCosineProduct
   
class EncoderTrainingModel(pl.LightningModule):
    """
    train definition for HSF
    """
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss weight
        self.cfg = cfg
        self.min_loss = float('inf')
        self._logger = logging.getLogger("HSF")

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        self.log('step_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        return loss.cpu().item()

    def validation_epoch_end(self, outputs) -> None:
        loss = np.mean([out for out in outputs])
        print(f'loss: {loss}')
        if loss < self.min_loss:
            self.min_loss = loss
            torch.save(self.state_dict(),
                       os.path.join(self.args.model_save_path, f'{self.__class__.__name__}_model.bin'))
            torch.save(self.model.state_dict(),
                       os.path.join('new_encoder', f'pytorch_model.bin'))
            print('model saved.')

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> None:
        self._logger.info('Test.')
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        scheduler = LambdaLR(optimizer,
                             lr_lambda=lambda step: min((step + 1) ** -0.5,
                                                        (step + 1) * self.args.warmup_epochs ** (-1.5)),
                             last_epoch=-1)
        return [optimizer], [scheduler]

class ClassifierTrainingModel(pl.LightningModule):
    """
    train definition for classifier
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # loss weight
        self.cfg = cfg
        self.min_loss = float('inf')
        self._logger = logging.getLogger("LLMeLog")

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        self.log('step_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        return loss.cpu().item()

    def validation_epoch_end(self, outputs) -> None:
        loss = np.mean([out for out in outputs])
        print(f'loss: {loss}')
        if loss < self.min_loss:
            self.min_loss = loss
            torch.save(self.state_dict(),
                       os.path.join(self.args.model_save_path, f'{self.__class__.__name__}_model.bin'))
            print('model saved.')

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> None:
        self._logger.info('Test.')
        self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        scheduler = LambdaLR(optimizer,
                             lr_lambda=lambda step: min((step + 1) ** -0.5,
                                                        (step + 1) * self.args.warmup_epochs ** (-1.5)),
                             last_epoch=-1)
        return [optimizer], [scheduler]
    

class HSFencoder(EncoderTrainingModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.args = cfg
        self.tokenizer = AutoTokenizer.from_pretrained("./bert-base-en/")
        self.model = AutoModel.from_pretrained("./bert-base-en")
        self.fc_anomal = MarginCosineProduct(768, self.args.anomal_class).to(self.args.hard_device)
        self.fc_nomal = MarginCosineProduct(768, 2).to(self.args.hard_device)
        self.loss = nn.CrossEntropyLoss()
        self.avg_pooling = nn.AdaptiveAvgPool2d((1,768))

    def forward(self, batch):
        nomal_src, anomal_src, nomal_label, anomal_label, anomal_class_label = batch
        if len(anomal_src) != 0:
            one_hot = torch.zeros([anomal_class_label.shape[0], self.args.anomal_class]).to(self.args.hard_device)
            one_hot.scatter_(1, anomal_class_label.view(-1, 1), 1.0)
            anomal_class_label = one_hot
        out = None
        loss = None
        labels = None
        if len(nomal_src) != 0:
            nomal_tgt = nomal_label.to(self.args.hard_device) 
            nomal_inputs = self.tokenizer(nomal_src, return_tensors="pt", padding = True).to(self.args.hard_device)
            nomal_output_pooler = self.model(**nomal_inputs)[1]
            nomal_out = self.fc_nomal(nomal_output_pooler, nomal_tgt)   
            loss = self.loss(nomal_out, nomal_tgt) * 1.5 
            out = nomal_output_pooler
            labels = nomal_label


        if len(anomal_src) != 0:
            anomal_tgt = anomal_label.to(self.args.hard_device) 
            anomal_class_tgt = anomal_class_label.to(self.args.hard_device)
            anomal_inputs = self.tokenizer(anomal_src, return_tensors="pt", padding = True).to(self.args.hard_device)
            anomal_output_pooler = self.model(**anomal_inputs)[1]
            anomal_out = self.fc_nomal(anomal_output_pooler, anomal_tgt)
            anomal_class_out = self.fc_anomal(anomal_output_pooler, anomal_class_tgt)
            anomal_loss = self.loss(anomal_out, anomal_tgt) * 1.5 + self.loss(anomal_class_out, anomal_class_tgt)
            if loss == None: 
                loss = anomal_loss
                out = anomal_output_pooler
                labels = anomal_label   
            else: 
                loss += anomal_loss
                out = torch.cat((out, anomal_output_pooler))
                labels = torch.cat((labels, anomal_label))

        return loss, (out, labels)
    
class LLMeLog(ClassifierTrainingModel):
    def __init__(self, cfg, embedding_dict):
        super().__init__(cfg)
        self.args = cfg
        self.encoder = make_model()
        self.fc =  nn.Sequential(nn.Dropout(), nn.Linear(768, 32), nn.ReLU(), nn.Dropout(), nn.Linear(32, 2))
        self.loss = nn.CrossEntropyLoss()
        self.embedding_dict = embedding_dict
        self.avg_pooling = nn.AdaptiveAvgPool2d((1,768))

    def deal_batch(self, batch):
        res_src, res_label = batch

        mask_src = np.array(res_src, dtype = float)
        mask_src = torch.from_numpy(mask_src)
        src_mask = (mask_src != 0).unsqueeze(-2)

        src_embedding = []
        
        for sent in res_src:
            sent_emd = [self.embedding_dict[str(word)] for word in sent]
            src_embedding.append(sent_emd)    

        src_embedding = np.array(src_embedding)
        src_embedding = torch.from_numpy(src_embedding).float()

        res_label = np.array(res_label)
        res_label = torch.from_numpy(res_label).float()

        return src_embedding.to(self.args.hard_device), res_label.to(self.args.hard_device), src_mask.to(self.args.hard_device)       

    def forward(self, batch):
        src, tgt, src_mask = self.deal_batch(batch) 
        out = self.encoder(src, src_mask)
        pooling_out = self.avg_pooling(out).squeeze(1)
        class_out = self.fc(pooling_out)
        loss = self.loss(class_out, tgt)
        return loss, class_out
