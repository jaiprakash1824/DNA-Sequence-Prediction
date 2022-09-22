from turtle import update
import pytorch_lightning as pl
import pandas as pd 
from typing import Optional
from torch.nn.utils.rnn import pad_sequence
import torch 
import pickle
from torch.utils.data import DataLoader

class DNASeqDataset(pl.LightningDataModule):
    def __init__(self, datapath = 'data/deepspeech.csv', device='cpu'):
        super().__init__()
        self.datapath = datapath
        self.signal_path = 'data/padded_signals'
        self.transcript_path = 'data/padded_transcripts'
        self.device = device 


    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage)

    def prepare_data(self) -> None:
        
        self.padded_signal = torch.load(self.signal_path)
        self.transcription = torch.load(self.transcript_path)

        self.padded_signal = self.padded_signal.to(torch.long).to(self.device)
        self.transcription = self.transcription.to(torch.long).to(self.device)
        

        with open('data/val2id.pkl', 'rb') as f:
            self.val2id = pickle.load(f)
            print(len(self.val2id.keys()))
        with open('data/t2id.pkl', 'rb') as f:
            self.t2id = pickle.load(f)
            print(len(self.t2id.keys()))

        self.data = [(signal, transcript) for signal, transcript in zip(self.padded_signal, self.transcription)]
    
    def train_dataloader(self):
        return DataLoader(self.data[:40000], batch_size=64)
    
    def val_dataloader(self):
        return DataLoader(self.data[40000:45000], batch_size=64)

    def test_dataloader(self):
        return DataLoader(self.data[45000:], batch_size=64)

if __name__ == '__main__':
    dna_seq_dataset = DNASeqDataset()
    dna_seq_dataset.prepare_data()
    trainloader = dna_seq_dataset.train_dataloader()
    src, tgt = iter(trainloader).next()
    print(src.size(), tgt.size())
    print(iter(trainloader).next())