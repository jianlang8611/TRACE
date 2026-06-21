import copy
import json
import os
import time
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from einops import rearrange
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, BertModel
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch_geometric.utils import dense_to_sparse

from .coattention import CoAttention
from ..Base.utils import orthogonal_loss, l2_loss_fn, infoNCE_align_loss, kl_divergence
from ..trace import ASRRefiner, GradientReversal, EventConfounderHead, compute_iepa_loss, compute_asr_align_loss

 


class AddLinear(nn.Module):
    def __init__(self, fea_dim=256):
        super(AddLinear, self).__init__()
        self.linear = nn.LazyLinear(fea_dim)
        
    def forward(self, x):
        return self.linear(x)

class SVFEND(nn.Module):

    def __init__(self,loss_type, asr_alpha=1, iepa_alpha=1, cfus_alpha=0.01,encoder_name='bert-base-uncased', fea_dim=128, dropout=0.1, ori=False, ablation='No', **kargs):

        super(SVFEND, self).__init__()

        self.text_dim = 768
        self.comment_dim = 768
        self.img_dim = 4096
        self.video_dim = 4096
        self.num_frames = 83
        self.num_audioframes = 50
        self.num_comments = 23
        self.dim = fea_dim
        self.num_heads = 4
        self.audio_dim = 128
        self.ori = ori
        self.dropout = dropout   
        self.ablation = ablation
        print(f"model ablation: {ablation}")

        # self.vggish_layer = torch.hub.load('models/torchvggish', 'vggish', source='local')
        # net_structure = list(self.vggish_layer.children())      
        # self.vggish_modified = nn.Sequential(*net_structure[-2:-1])
        # # freeze vggish
        # for param in self.vggish_modified.parameters():
        #     param.requires_grad = False
        
        self.vggish_modified = None
        
        self.co_attention_ta = CoAttention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout, d_model=fea_dim,
                                    visual_len=self.num_audioframes, sen_len=512, fea_v=self.dim, fea_s=self.dim, pos=False)
        self.co_attention_tv = CoAttention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout, d_model=fea_dim,
                                    visual_len=self.num_frames, sen_len=512, fea_v=self.dim, fea_s=self.dim, pos=False)

        self.trm = nn.TransformerEncoderLayer(d_model=self.dim, nhead=2, dropout=dropout, batch_first=True)

        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_comment = nn.Sequential(torch.nn.Linear(self.comment_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_video = nn.Sequential(torch.nn.Linear(self.video_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_intro = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim),torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_audio = nn.Sequential(torch.nn.Linear(self.audio_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))

        if not self.ori:
            rho = kargs.get('rho', 0.9)
            self.asr_refiner_text = ASRRefiner(fea_dim=fea_dim, rho=rho)
            self.asr_refiner_audio = ASRRefiner(fea_dim=fea_dim, rho=rho)
            self.asr_refiner_img = ASRRefiner(fea_dim=fea_dim, rho=rho)
            self.asr_refiner_video = ASRRefiner(fea_dim=fea_dim, rho=rho)
            self.asr_refiner_intro = ASRRefiner(fea_dim=fea_dim, rho=rho)
            self.asr_refiner_comment = ASRRefiner(fea_dim=fea_dim, rho=rho)
        self.classifier = nn.Linear(fea_dim,2)
        if not self.ori:
            self.num_events = kargs.get('num_events', 1000)
            self.event_head = EventConfounderHead(
                input_dim=fea_dim * 4,
                num_events=self.num_events,
                lambda_grl=kargs.get('event_grl_lambda', 1.0),
                hidden_dim=256,
                dropout=0.5
            )

        self.ablation = ablation
        self.loss_type = loss_type
        self.asr_alpha = asr_alpha
        self.iepa_alpha = iepa_alpha
        self.cfus_alpha = cfus_alpha
        self.diric_alpha = kargs.get('diric_alpha', 1.0)

    def forward(self, **kwargs):

        if 'event' in kwargs:
            events = kwargs['event']
        else:
            events = None

        intro_fea = kwargs['intro_fea']
        fea_intro = self.linear_intro(intro_fea)
        comment_fea = kwargs['comment_fea']
        fea_comment = self.linear_comment(comment_fea)
        text_fea = kwargs['text_fea']
        fea_text = self.linear_text(text_fea) 

        audioframes=kwargs['audioframes']
        fea_audio = self.linear_audio(audioframes)

        frames=kwargs['frames']
        fea_img = self.linear_img(frames)
        


        c3d = kwargs['c3d']
        fea_video = self.linear_video(c3d)

        unrefined_fea_text = fea_text
        unrefined_fea_audio = fea_audio
        unrefined_fea_img = fea_img
        unrefined_fea_video = fea_video
        unrefined_fea_intro = fea_intro
        unrefined_fea_comment = fea_comment
        if not self.ori and self.ablation != 'w/o-asr':

            refined_fea_text, S_text, m_text = self.asr_refiner_text(fea_text, events)
            refined_fea_audio, S_audio, m_audio = self.asr_refiner_audio(fea_audio, events)
            refined_fea_img, S_img, m_img = self.asr_refiner_img(fea_img, events)
            refined_fea_video, S_video, m_video = self.asr_refiner_video(fea_video, events)
            refined_fea_intro, S_intro, m_intro = self.asr_refiner_intro(fea_intro, events)
            refined_fea_comment, S_comment, m_comment = self.asr_refiner_comment(fea_comment, events)

        else:
            refined_fea_text, S_text, m_text = fea_text, torch.zeros_like(fea_text), torch.zeros_like(fea_text)
            refined_fea_audio, S_audio, m_audio = fea_audio, torch.zeros_like(fea_audio), torch.zeros_like(fea_audio)
            refined_fea_img, S_img, m_img = fea_img, torch.zeros_like(fea_img), torch.zeros_like(fea_img)
            refined_fea_video, S_video, m_video = fea_video, torch.zeros_like(fea_video), torch.zeros_like(fea_video)
            refined_fea_intro, S_intro, m_intro = fea_intro, torch.zeros_like(fea_intro), torch.zeros_like(fea_intro)
            refined_fea_comment, S_comment, m_comment = fea_comment, torch.zeros_like(fea_comment), torch.zeros_like(fea_comment)



        fea_audio, fea_text = self.co_attention_ta(v=fea_audio, s=fea_text, v_len=fea_audio.shape[1], s_len=fea_text.shape[1])
        fea_audio = torch.mean(fea_audio, -2)

        fea_img, fea_text = self.co_attention_tv(v=fea_img, s=fea_text, v_len=fea_img.shape[1], s_len=fea_text.shape[1])
        fea_img = torch.mean(fea_img, -2)

        fea_text = torch.mean(fea_text, -2)      
        fea_video = torch.mean(fea_video, -2)



        if not self.ori:
            def _pool_to_vector(x):
                return torch.mean(x, dim=1) if x.dim() == 3 else x
            ev_text = _pool_to_vector(unrefined_fea_text)
            ev_audio = _pool_to_vector(unrefined_fea_audio)
            ev_img = _pool_to_vector(unrefined_fea_img)
            ev_video = _pool_to_vector(unrefined_fea_video)
            event_feat_concat = torch.cat([ev_text, ev_audio, ev_video, ev_img], dim=-1)
            event_logits = self.event_head(event_feat_concat)
        

        fea_text = fea_text.unsqueeze(1)
        fea_img = fea_img.unsqueeze(1)
        fea_audio = fea_audio.unsqueeze(1)
        fea_video = fea_video.unsqueeze(1)
        fea_intro = fea_intro.unsqueeze(1)
        fea_comment = fea_comment.unsqueeze(1)


        fea=torch.cat((fea_text,fea_audio, fea_video, fea_intro,fea_img, fea_comment),1)
        fea = self.trm(fea)
        tsne_tensor = fea.mean(1).clone()
        fea = torch.mean(fea, -2)
        
        cls_output = self.classifier(fea)
        output = cls_output

        ret_dict = {
            'pred': output,
            'cls_output': cls_output,
            'tsne_tensor': tsne_tensor,
            'ori': self.ori,
            'fea': fea,
            'event': events  if not self.ori else None,
            'event_logits': event_logits if not self.ori else None,
        }
        if not self.ori and self.ablation != 'w/o-asr':
            ret_dict['refined_fea_text'] = refined_fea_text
            ret_dict['refined_fea_audio'] = refined_fea_audio
            ret_dict['refined_fea_img'] = refined_fea_img
            ret_dict['refined_fea_video'] = refined_fea_video
            ret_dict['refined_fea_intro'] = refined_fea_intro
            ret_dict['refined_fea_comment'] = refined_fea_comment
            ret_dict['S_text'] = S_text
            ret_dict['S_audio'] = S_audio
            ret_dict['S_img'] = S_img
            ret_dict['S_video'] = S_video
            ret_dict['S_intro'] = S_intro
            ret_dict['S_comment'] = S_comment
            ret_dict['m_text'] = m_text
            ret_dict['m_audio'] = m_audio
            ret_dict['m_img'] = m_img
            ret_dict['m_video'] = m_video
            ret_dict['m_intro'] = m_intro
            ret_dict['m_comment'] = m_comment
            ret_dict['unrefined_fea_text'] = unrefined_fea_text
            ret_dict['unrefined_fea_audio'] = unrefined_fea_audio
            ret_dict['unrefined_fea_img'] = unrefined_fea_img
            ret_dict['unrefined_fea_video'] = unrefined_fea_video
            ret_dict['unrefined_fea_intro'] = unrefined_fea_intro
            ret_dict['unrefined_fea_comment'] = unrefined_fea_comment
        return ret_dict
    
    def cal_loss(self, **kwargs):

        label = kwargs['label']
        cls_output = kwargs['cls_output']
        loss = torch.tensor(0.0, device=label.device)
        align_loss = torch.tensor(0.0, device=label.device)
        iepa_klloss = torch.tensor(0.0, device=label.device)
        confusion_loss = torch.tensor(0.0, device=label.device)
        cls_loss = F.cross_entropy(cls_output, label)
        
        loss += cls_loss

        if not self.ori and self.ablation != 'w/o-asr':
            event_logits = kwargs['event_logits']
            event_labels = kwargs['event']
            confusion_loss = self.cfus_alpha * F.cross_entropy(event_logits, event_labels)
            loss += confusion_loss
        if not self.ori and self.ablation != 'w/o-asr':
            refined_list = [
                kwargs['refined_fea_text'],
                kwargs['refined_fea_audio'],
                kwargs['refined_fea_img'],
                kwargs['refined_fea_video'],
                kwargs['refined_fea_intro'],
                kwargs['refined_fea_comment'],
            ]
            unrefined_list = [
                kwargs['unrefined_fea_text'],
                kwargs['unrefined_fea_audio'],
                kwargs['unrefined_fea_img'],
                kwargs['unrefined_fea_video'],
                kwargs['unrefined_fea_intro'],
                kwargs['unrefined_fea_comment'],
            ]
            align_loss = compute_asr_align_loss(
                unrefined_list=unrefined_list,
                refined_list=refined_list,
                weight=self.asr_alpha
            )
            loss += align_loss
        if not self.ori and self.ablation != 'w/o-iepa':
            iepa_kl_total = compute_iepa_loss(
                logits=cls_output,
                labels=label,
                events=kwargs['event'],
                alpha=float(self.diric_alpha),
                num_classes=cls_output.size(-1)
            )
            iepa_klloss += iepa_kl_total * self.iepa_alpha
            loss += iepa_klloss
        return loss, cls_loss, align_loss, iepa_klloss, confusion_loss