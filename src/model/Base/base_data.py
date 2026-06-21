from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from pathlib import Path


class Base_Dataset(Dataset):
    def __init__(self, **kargs):
        super().__init__()
        self.data_path = Path('datasets')

    def _get_data(self, fold: int, split: str, task: str='binary'):
        raise NotImplementedError

class FakeSV_Dataset(Base_Dataset):
    def __init__(self, **kargs):
        super(FakeSV_Dataset, self).__init__()
        self.data_path = Path('datasets/FakeSV')
    
    def _get_complete_data(self):
        data_complete = pd.read_json('datasets/FakeSV/data_complete.jsonl', orient='records', dtype=False, lines=True)
        replace_values = {'辟谣': 2, '假': 1, '真':0}
        data_complete['label'] = data_complete['annotation'].replace(replace_values)
        data_complete = data_complete[data_complete['label']!=2]
        data_complete['event'], _ = pd.factorize(data_complete['keywords'])
        data_complete['vid'] = data_complete['video_id']
        return data_complete
    
    def _get_data(self, fold, split, task='binary'):
        if fold in [1, 2, 3, 4, 5]:
            data = self._get_fold_data(fold, split)
        elif fold in ['temporal', 'default']:
            data = self._get_temporal_data(split)
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data
    
    def _get_fold_data(self, fold, split):
        if split == 'train':
            vid_path = f'datasets/FakeSV/vids/vid_fold_no_{fold}.txt'
        elif split == 'test':
            vid_path = f'datasets/FakeSV/vids/vid_fold_{fold}.txt'
        else:
            raise ValueError(f"Invalid split: {split}")
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data

    def _get_temporal_data(self, split: str):
        vid_path = f'datasets/FakeSV/vids/vid_time3_{split}.txt'
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['video_id'].isin(vids)]
        return data


class FakeTT_Dataset(Base_Dataset):
    def __init__(self, **kargs):
        super(FakeTT_Dataset, self).__init__()
        self.data_path = Path('datasets/FakeTT')
    
    def _get_complete_data(self):
        data = pd.read_json('datasets/FakeTT/data_complete.jsonl', orient='records', lines=True, dtype={'video_id': 'str'})
        replace_values = {'fake': 1, 'real': 0}
        data['label'] = data['annotation'].replace(replace_values)
        data['event'], _ = pd.factorize(data['event'])
        data['vid'] = data['video_id']
        title_data = pd.read_json('datasets/FakeTT/data.jsonl', orient='records', lines=True, dtype={'vid': 'str'})
        vid_to_title = dict(zip(title_data['vid'], title_data['title']))
        data['title'] = data['vid'].map(vid_to_title)
        if data['title'].isna().any():
            missing_vids = data[data['title'].isna()]['vid'].tolist()
            raise ValueError(f"not found title for {missing_vids}")
        # set type of video_id to str
        return data
    
    def _get_data(self, fold, split, task='binary'):
        if fold in ['temporal', 'default']:
            data = self._get_temporal_data(split)
        elif fold in [1, 2, 3, 4, 5]:
            data = self._get_fold_data(fold, split)
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data
    
    def _get_fold_data(self, fold, split):
        if split == 'train':
            vid_path = f'datasets/FakeTT/vids/vid_fold_no_{fold}.txt'
        elif split == 'test':
            vid_path = f'datasets/FakeTT/vids/vid_fold_{fold}.txt'
        else:
            raise ValueError(f"Invalid split: {split}")
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['vid'].isin(vids)]
        return data

    def _get_temporal_data(self, split: str):
        vid_path = f'datasets/FakeTT/vids/vid_time3_{split}.txt'
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['vid'].isin(vids)]
        return data


class FVC_Dataset(Base_Dataset):
    def __init__(self, **kargs):
        super(FVC_Dataset, self).__init__()
        self.data_path = Path('datasets/FVC')
    
    def _get_complete_data(self):
        data = pd.read_json('datasets/FVC/data.jsonl', orient='records', lines=True, dtype={'vid': 'str'})
        desc_df = pd.read_json('datasets/FVC/data_complete.jsonl', orient='records', lines=True, dtype={'vid': 'str'})[['vid', 'description']]
        data = data.merge(desc_df, on='vid', how='left')
        if 'description' not in data.columns:
            data['description'] = ''
        data['description'] = data['description'].fillna('')
        data['event'], _ = pd.factorize(data['event_id'])
        return data
    
    def _get_data(self, fold, split, task='binary'):
        if fold in ['temporal', 'default']:
            data = self._get_temporal_data(split)
        elif fold in [1, 2, 3, 4, 5]:
            data = self._get_fold_data(fold, split)
        else:
            raise NotImplementedError(f"Invalid fold: {fold}")
        return data
    
    def _get_fold_data(self, fold, split):
        if split == 'train':
            vid_path = f'datasets/FVC/vids/vid_fold_no_{fold}.txt'
        elif split == 'test':
            vid_path = f'datasets/FVC/vids/vid_fold_{fold}.txt'
        else:
            raise ValueError(f"Invalid split: {split}")
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['vid'].isin(vids)]
        return data

    def _get_temporal_data(self, split: str):
        vid_path = f'datasets/FVC/vids/vid_time3_{split}.txt'
        with open(vid_path, "r") as fr:
            vids = [line.strip() for line in fr.readlines()]
        data = self._get_complete_data()
        data = data[data['vid'].isin(vids)]
        return data