from turtle import update
import pytorch_lightning as pl
import pandas as pd 
from typing import Optional
from torch.nn.utils.rnn import pad_sequence
import torch 
import pickle

class DNASeqDataset(pl.LightningDataModule):
    def __init__(self, datapath = 'data/deepspeech.csv'):
        super().__init__()
        self.datapath = datapath




    def setup(self, stage: Optional[str] = None) -> None:
        return super().setup(stage)

    def prepare_data(self) -> None:
        df = pd.read_csv(self.datapath, converters={'signal': pd.eval})
        signals = df['signal'].tolist()
        transcripts = df['transcript'].tolist()
        t2id = {'A':1, 'C':2, 'G':3, 'T':4}
        unique_signal_tokens = set()

        for signal in signals:
            for signal_value in signal:
                unique_signal_tokens.add(signal_value)
        val2id = {val: id+1 for id, val in  enumerate(unique_signal_tokens)}

        updated_signals = []
        for signal in signals:
            updated_signal = []
            for signal_val in signal:
                updated_signal.append(val2id[signal_val])
            updated_signals.append(torch.Tensor(updated_signal))
        
        padded_signal_sequence = pad_sequence(updated_signals)
        print(padded_signal_sequence.size())


        
        # print(unique_signal_tokens)
        updated_transcripts = []
        for transcript in transcripts:
            updated_transcript = []
            for char in transcript:
                updated_transcript.append(t2id[char])
            updated_transcripts.append(torch.Tensor(updated_transcript))
        
        padded_transcript_sequence = pad_sequence(updated_transcripts)
        print(padded_transcript_sequence.size())
        torch.save(torch.t(padded_signal_sequence), 'data/padded_signals')
        torch.save(torch.t(padded_transcript_sequence), 'data/padded_transcripts')

        with open('data/val2id.pkl', 'wb') as f:
            pickle.dump(val2id, f)
        with open('data/t2id.pkl', 'wb') as f:
            pickle.dump(t2id, f)







        return super().prepare_data()
    
    def train_dataloader(self):
        return super().train_dataloader()
    
    def val_dataloader(self):
        return super().val_dataloader()

    def test_dataloader(self):
        return super().test_dataloader()

if __name__ == '__main__':
    dna_seq_dataset = DNASeqDataset()
    dna_seq_dataset.prepare_data()