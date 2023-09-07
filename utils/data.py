import os
import torch
from torch import nn
import json
from tqdm import tqdm, trange
from typing import List
from transformers import AutoTokenizer
from collections import defaultdict
import re
from pdb import set_trace as stop


EMO_MAP = {
    -1: 1,
    0: 2,
    1: 3,
    2: 4
}

def pass_offset(data_path, offset):
    """
    Set the offset in english dataset(Camera-COQE) starts from 0.
    """
    if 'Camera' in data_path:
        return offset - 1
    else:
        return pass_offset

def proc_raw_offset(offset_spans: str, text, data_path):
    if offset_spans == '':
        # use 1 to denotes the empty span
        return (0, 0)
    # 7&&all 8&&of 9&&the 10&&Nikon 11&&DLSR 12&&models
    if 'Camera' in data_path:
        offsets = re.findall(r'([0-9]+)&&(\S+)', offset_spans)
    else:
        offsets = re.findall(r'([0-9]+)&(\S+)', offset_spans)
    # [7&&all, 8&&of, 9&&the, 10&&Nikon, 11&&DLSR, 12&&models]
    return int(offsets[0][0]), int(offsets[-1][0])


def process_line(args, text_line, label_line, tokenizer: AutoTokenizer, sample_id):
    text = text_line.split('\t')[0].strip()
    have_triples = int(text_line.split('\t')[1])

    re_result = re.findall(r'\[\[(.*?)\];\[(.*?)\];\[(.*?)\];\[(.*?)\];\[(.*?)\]\]', label_line)
    raw_labels: List = [[x for x in y] for y in re_result]
    # List of triples for a sentence

    tokens_output = tokenizer(text, max_length=args.max_text_length - 1, pad_to_max_length=True)
    token_ids = [tokenizer.convert_tokens_to_ids('[unused1]')] + tokens_output['input_ids']
    sample = {'token_ids': token_ids, 'labels': [], 'sample_id': sample_id}
    if have_triples == 0:
        return sample


    for tri in raw_labels:
        # tri: [sub, obj, aspect, opinion, sentiment]
        sub_offset = proc_raw_offset(tri[0], text, args.data_path)
        obj_offset = proc_raw_offset(tri[1], text, args.data_path)
        aspect_offset = proc_raw_offset(tri[2], text, args.data_path)
        view_offset = proc_raw_offset(tri[3], text, args.data_path)

        sentiment_label = have_triples * EMO_MAP[int(tri[4])]

        if 'Camera' in args.data_path:
            sample['labels'].append({
                'sub_start_index': tokens_output.word_to_tokens(sub_offset[0]).start,
                'sub_end_index': tokens_output.word_to_tokens(sub_offset[1]).end - 1,
                'obj_start_index': tokens_output.word_to_tokens(obj_offset[0]).start,
                'obj_end_index': tokens_output.word_to_tokens(obj_offset[1]).end - 1,
                'aspect_start_index': tokens_output.word_to_tokens(aspect_offset[0]).start,
                'aspect_end_index': tokens_output.word_to_tokens(aspect_offset[1]).end - 1,
                'opinion_start_index': tokens_output.word_to_tokens(view_offset[0]).start,
                'opinion_end_index': tokens_output.word_to_tokens(view_offset[1]).end - 1,
                'relation': sentiment_label
            })
        else:
            try:
                sample['labels'].append({
                    'sub_start_index': tokens_output.char_to_token(sub_offset[0]) + 1,
                    'sub_end_index': tokens_output.char_to_token(sub_offset[1]) + 1,
                    'obj_start_index': tokens_output.char_to_token(obj_offset[0]) + 1,
                    'obj_end_index': tokens_output.char_to_token(obj_offset[1]) + 1,
                    'aspect_start_index': tokens_output.char_to_token(aspect_offset[0]) + 1,
                    'aspect_end_index': tokens_output.char_to_token(aspect_offset[1]) + 1,
                    'opinion_start_index': tokens_output.char_to_token(view_offset[0]) + 1,
                    'opinion_end_index': tokens_output.char_to_token(view_offset[1]) + 1,
                    'relation': sentiment_label
                })
            except:
                print("zhijiang datasets error")
                stop()
    # stop()
    return sample
        
def load_data(args, mode: str):
    raw_data = []
    with open(os.path.join(args.data_path, f'{mode}.txt'), 'r') as f:
        for line in f:
            raw_data.append(line)
    all_samples = []
    line_id, i = 0, 0
    text_line, label_line = '', ''
    for line_id in trange(len(raw_data), desc=f'processing data for mode {mode}'):
        cur_line = raw_data[line_id]
        if len(cur_line.split('\t')) != 2:
            label_line += '\n' + cur_line
        else:
            # a new text line, so push the last text and update text_line
            if text_line != '':
                all_samples.append(process_line(args, text_line, label_line, args.tokenizer, i))
                i += 1
            text_line = cur_line
            label_line = ''
    
    all_samples.append(process_line(args, text_line, label_line, args.tokenizer, i))
    
    return all_samples

def build_collate_fn(args):
    def collate_fn(batch):
        input_ids = torch.tensor([sample['token_ids'] for sample in batch], device=args.device, dtype=torch.long)
        seq_ids = [sample['sample_id'] for sample in batch]
        labels = []
        for sample in batch:
            target = {
                'sub_start_index': [],
                'sub_end_index': [],
                'obj_start_index': [],
                'obj_end_index': [],
                'aspect_start_index': [],
                'aspect_end_index': [],
                'opinion_start_index': [],
                'opinion_end_index': [],
                'relation': [],
            }
            for tri in sample['labels']:
                for k in tri:
                    target[k].append(tri[k])

            for k in target:
                assert len(target[k]) <= args.num_generated_triples
                target[k] = torch.tensor(target[k], device=args.device, dtype=torch.long)
            labels.append(target)
        return input_ids, labels, seq_ids
    return collate_fn