import pandas as pd
import torch
from transformers import AutoTokenizer, AutoProcessor
from pathlib import Path
from PIL import Image
import os
import numpy as np

from ..Base.base_data import Base_Dataset, FakeSV_Dataset, FakeTT_Dataset, FVC_Dataset


_feature_cache = {}
def load_feature(path):
    if path not in _feature_cache:
        # print(f"Loading feature from {path}...")
        _feature_cache[path] = torch.load(path, weights_only=True)
    return _feature_cache[path]


class SVFEND_Dataset(Base_Dataset):
    def __init__(self, fold: int, split: str, task: str, **kargs):
        super().__init__()
        self.fea_path = self.data_path / 'fea' / 'SVFEND'    

        self.vggishfeapath = self.fea_path / 'vggish_128d_features_bn.pt'  # (batch, 36, 128)
        # self.vggishfeapath = self.fea_path / 'vggish_128d_features.pt'
        # self.vggishfeapath = self.fea_path / 'vggish_pre_features.pt'
        # self.framefeapath= self.fea_path / 'vgg19_features_bn.pt' # (batch, 32, dim)
        self.framefeapath= self.fea_path / 'vgg19_features.pt'

        self.c3dfeapath= self.fea_path / 'c3d_features.pt'
        # self.textfeapath = self.fea_path / 'fea_text_bn.pt'
        self.textfeapath = self.fea_path / 'fea_text_2.pt'
        self.introfeapath = self.fea_path / 'fea_intro.pt' # use intro & comment on FakeSV
        self.commentfeapath = self.fea_path / 'fea_comments.pt'
        self.data = self._get_data(fold, split, task)
        # self.data['description'] = self.data['title']

        self.frame_fea = load_feature(self.framefeapath)
        self.c3d_fea = load_feature(self.c3dfeapath)
        self.vggish_fea = load_feature(self.vggishfeapath)
        self.text_fea = load_feature(self.textfeapath)
        self.intro_fea = load_feature(self.introfeapath)
        self.comment_fea = load_feature(self.commentfeapath)
    
    def __len__(self):
        return len(self.data)
     
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['vid']
        label = item['label']

        audioframes = self.vggish_fea[vid]
        frames = self.frame_fea[vid]
        if frames.dim() == 1:
            if frames.numel() == 4096:
                frames = frames.unsqueeze(0)
            elif frames.numel() % 4096 == 0:
                frames = frames.view(-1, 4096)
            else:
                frames = frames.unsqueeze(0)
        c3d = self.c3d_fea[vid]
        if c3d.dim() == 1:
            if c3d.numel() == 4096:
                c3d = c3d.unsqueeze(0)
            elif c3d.numel() % 4096 == 0:
                c3d = c3d.view(-1, 4096)
            else:
                c3d = c3d.unsqueeze(0)
        text_fea = self.text_fea[vid]
        intro_fea = self.intro_fea[vid]
        comment_fea = self.comment_fea[vid]
        
        return {
            'vid': vid,
            'label': torch.tensor(label),
            'audioframes': audioframes,
            'frames':frames,
            'c3d': c3d,
            'text_fea': text_fea,
            'intro_fea': intro_fea,
            'comment_fea': comment_fea,
            'event': torch.tensor(item['event']),
        }

class SVFEND_Collator:
    def __init__(self, **kargs):
        pass
    def __call__(self, batch):
        num_frames = 83
        num_audioframes = 50 
        
        vids = [item['vid'] for item in batch]
        
        text_fea = [item['text_fea'] for item in batch]
        text_fea = torch.stack(text_fea)
        intro_fea = [item['intro_fea'] for item in batch]
        intro_fea = torch.stack(intro_fea)
        comment_fea = [item['comment_fea'] for item in batch]
        comment_fea = torch.stack(comment_fea)
        frames = [item['frames'] for item in batch]
        frames, frames_masks = pad_frame_sequence(num_frames, frames)
        # frames = torch.stack(frames)

        audioframes  = [item['audioframes'] for item in batch]
        # audioframes = torch.stack(audioframes)
        audioframes, audioframes_masks = pad_frame_sequence(num_audioframes, audioframes)

        c3d = [item['c3d'] for item in batch]
        # c3d = torch.stack(c3d)
        # _, c3d_masks = pad_frame_sequence(num_frames, c3d)
        c3d, c3d_masks = pad_frame_sequence(num_frames, c3d)

        labels = [item['label'] for item in batch]
        labels = torch.stack(labels)
        
        events = [item['event'] for item in batch]
        events = torch.stack(events)
        
        return {
            'vids': vids,
            'labels': labels,
            'text_fea': text_fea,
            'intro_fea': intro_fea,
            'comment_fea': comment_fea,
            'audioframes': audioframes,
            'frames':frames,
            'c3d': c3d,
            'event': events,
        }
        
class FakeSV_SVFEND_Dataset(SVFEND_Dataset, FakeSV_Dataset):
    def __init__(self, fold: int, split: str, task: str, **kargs):
        super().__init__(fold=fold, split=split, task=task, **kargs)
        
class FakeSV_SVFEND_Collator(SVFEND_Collator):
    pass

        
class FakeTT_SVFEND_Dataset(SVFEND_Dataset, FakeTT_Dataset):
    def __init__(self, fold: int, split: str, task: str, **kargs):
        super().__init__(fold=fold, split=split, task=task, **kargs)

class FakeTT_SVFEND_Collator(SVFEND_Collator):
    pass


class FVC_SVFEND_Dataset(SVFEND_Dataset, FVC_Dataset):
    def __init__(self, fold: int, split: str, task: str, **kargs):
        super().__init__(fold=fold, split=split, task=task, **kargs)

class FVC_SVFEND_Collator(SVFEND_Collator):
    pass


def pad_frame_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)