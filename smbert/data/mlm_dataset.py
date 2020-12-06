import math
import random
import pkuseg
import numpy as np

from tqdm import tqdm
from smbert.common.tokenizers import Tokenizer
from pretrain_config import *
from torch.utils.data import Dataset


class DataFactory(object):
    def __init__(self):
        self.tokenizer = Tokenizer(VocabPath)
        self.seg = pkuseg.pkuseg()
        self.vocab_size = self.tokenizer._vocab_size
        self.token_pad_id = self.tokenizer._token_pad_id
        self.token_cls_id = self.tokenizer._token_start_id
        self.token_sep_id = self.tokenizer._token_end_id
        self.token_mask_id = self.tokenizer._token_mask_id

    def __token_process(self, token_id):
        """
        以80%的几率替换为[MASK]，以10%的几率保持不变，
        以10%的几率替换为一个随机token。
        """
        rand = np.random.random()
        if rand <= 0.8:
            return self.token_mask_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0, self.vocab_size)

    def texts_to_ids(self, texts):
        texts_ids = []
        for text in texts:
            # 处理每个句子
            for word in text:
                # text_ids首位分别是cls和sep，这里暂时去除
                word_tokes = self.tokenizer.tokenize(text=word)[1:-1]
                words_ids = self.tokenizer.tokens_to_ids(word_tokes)
                texts_ids.append(words_ids)
        return texts_ids

    def ids_to_mask(self, texts_ids):
        instances = []
        total_ids = []
        total_masks = []
        # 为每个字或者词生成一个概率，用于判断是否mask
        mask_rates = np.random.random(len(texts_ids))

        for i, word_id in enumerate(texts_ids):
            # 为每个字生成对应概率
            total_ids.extend(word_id)
            if mask_rates[i] < MaskRate:
                # 因为word_id可能是一个字，也可能是一个词
                for sub_id in word_id:
                    total_masks.append(self.__token_process(sub_id))
            else:
                total_masks.extend([0]*len(word_id))

        # 每个实例的最大长度为512，因此对一个段落进行裁剪
        # 510 = 512 - 2，给cls和sep留的位置
        for i in range(math.ceil(len(total_ids)/(SentenceLength - 2))):
            tmp_ids = [self.token_cls_id]
            tmp_masks = [self.token_pad_id]
            tmp_ids.extend(total_ids[i*(SentenceLength - 2): min((i+1)*(SentenceLength - 2), len(total_ids))])
            tmp_masks.extend(total_masks[i*(SentenceLength - 2): min((i+1)*(SentenceLength - 2), len(total_masks))])
            # 不足512的使用padding补全
            diff = SentenceLength - len(tmp_ids)
            if diff == 1:
                tmp_ids.append(self.token_sep_id)
                tmp_masks.append(self.token_pad_id)
            else:
                # 添加结束符
                tmp_ids.append(self.token_sep_id)
                tmp_masks.append(self.token_pad_id)
                # 将剩余部分padding补全
                tmp_ids.extend([self.token_pad_id] * (diff - 1))
                tmp_masks.extend([self.token_pad_id] * (diff - 1))
            instances.append([tmp_ids, tmp_masks])
        return instances

    def ids_all_mask(self, texts_ids):
        instances = []
        tmp_ids = [101]

        # 格式化数据
        for token_ids in texts_ids:
            if isinstance(token_ids, list):
                for token_id in token_ids:
                    tmp_ids.append(token_id)
                    if len(tmp_ids) == SentenceLength - 1:
                        break
            else:
                tmp_ids.append(token_ids)
                if len(tmp_ids) == SentenceLength - 1:
                    break
            if len(tmp_ids) == SentenceLength - 1:
                break

        tmp_ids.append(102)
        input_length = len(tmp_ids) - 2
        if len(tmp_ids) < SentenceLength:
            for i in range(SentenceLength - len(tmp_ids)):
                tmp_ids.append(0)

        return tmp_ids


class SMBertDataSet(Dataset):
    def __init__(self, corpus_path, onehot_type=False):
        self.corpus_path = corpus_path
        self.onehot_type = onehot_type
        self.smbert_data = DataFactory()
        self.src_lines , self.corr_lines , self.labels= [],[],[]
        self.tar_lines , self.corr_tar_lines = [] , []
        self.tokenid_to_count = {}
        for i in range(RepeatNum):
            for texts in tqdm(self.__get_texts()):
                texts_lis = str(texts).split('-***-')
                text = texts_lis[0].strip()
                correct_text = texts_lis[1].strip()
                label = texts_lis[2].strip()

                texts_ids = self.smbert_data.texts_to_ids(text)
                self.src_lines.append(texts_ids)
                correct_texts_ids = self.smbert_data.texts_to_ids(correct_text)
                self.corr_lines.append(correct_texts_ids)
                labels = [int(i) for i in label if i != ' ']
                labels = [0] + labels[:min(len(labels) , SentenceLength-2)] + [0]
                pad_label_len = SentenceLength - len(labels)
                labels = labels + [0]*pad_label_len
                self.labels.append(labels)

        for line in self.src_lines:
            instances = self.smbert_data.ids_all_mask(line)
            self.tar_lines.append(instances)
        for line in self.corr_lines:
            instances = self.smbert_data.ids_all_mask(line)
            self.corr_tar_lines.append(instances)

    def __get_texts(self):
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip()

    def __len__(self):
        return len(self.tar_lines)

    def __getitem__(self, item):
        output = {}
        input_token_ids = self.tar_lines[item]
        token_ids = self.corr_tar_lines[item]
        labels = self.labels[item]
        segment_ids = [0 if x else 0 for x in input_token_ids]
        output['input_token_ids'] = input_token_ids
        output['token_ids_labels'] = token_ids
        output['segment_ids'] = segment_ids
        output['label'] = labels

        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance

    def __gen_input_token(self, token_ids, mask_ids):
        assert len(token_ids) == len(mask_ids)
        input_token_ids = []
        for token, mask in zip(token_ids, mask_ids):
            if mask == 0:
                input_token_ids.append(token)
            else:
                input_token_ids.append(mask)
        return input_token_ids

    def __id_to_onehot(self, token_ids):
        onehot_labels = []
        onehot_pad = [0] * VocabSize
        onehot_pad[0] = 1
        for i in token_ids:
            tmp = [0 for j in range(VocabSize)]
            if i == 0:
                onehot_labels.append(onehot_pad)
            else:
                tmp[i] = 1
                onehot_labels.append(tmp)
        return onehot_labels

    def __maskid_to_onehot(self, token_ids, is_masked):
        onehot_masked_labels = []
        for i in is_masked:
            onehot_labels = [0] * VocabSize
            onehot_labels[token_ids[i]] = 1
            onehot_masked_labels.append(onehot_labels)
        return onehot_masked_labels


class RobertaTestSet(Dataset):
    def __init__(self, test_path):
        self.tokenizer = Tokenizer(VocabPath)
        self.test_path = test_path
        self.test_lines = []
        self.label_lines = []
        self.labels = []
        # 读取数据
        with open(self.test_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line_list = line.strip().split('-***-')

                    self.test_lines.append(line_list[0].strip())
                    self.label_lines.append(line_list[1].strip())
                    label = line_list[2].strip()
                    labels = [int(i) for i in label if i != ' ']
                    labels = [0] + labels[:min(len(labels), SentenceLength - 2)] + [0]
                    pad_label_len = SentenceLength - len(labels)
                    labels = labels + [0] * pad_label_len
                    self.labels.append(labels)

    def __len__(self):
        return len(self.label_lines)

    def __getitem__(self, item):
        output = {}
        test_text = self.test_lines[item]
        label_text = self.label_lines[item]
        labels = self.labels[item]
        test_token = self.__gen_token(test_text)
        label_token = self.__gen_token(label_text)
        segment_ids = [0 if x else 0 for x in label_token]
        output['input_token_ids'] = test_token
        output['token_ids_labels'] = label_token
        output['segment_ids'] = segment_ids
        output['label'] = labels
        instance = {k: torch.tensor(v, dtype=torch.long) for k, v in output.items()}
        return instance

    def __gen_token(self, tokens):
        tar_token_ids = [101]
        tokens = list(tokens)
        tokens = tokens[:(SentenceLength - 2)]
        for token in tokens:
            token_id = self.tokenizer.token_to_id(token)
            tar_token_ids.append(token_id)
        tar_token_ids.append(102)
        if len(tar_token_ids) < SentenceLength:
            for i in range(SentenceLength - len(tar_token_ids)):
                tar_token_ids.append(0)
        return tar_token_ids
