import time
import torch

cuda_condition = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_condition else 'cpu')

# ## mlm模型文件路径 ## #
PronunciationPath = '../data/char_meta.txt'
path_train_data = '../data/train_data/train.sgml'
path_train_save = '../data/train_data/train.txt'
path_dev_data = '../data/test_data/train15.sgml'
path_dev_save = '../data/test_data/test.txt'

# Debug开关
Debug = False

# 使用预训练模型开关
UsePretrain = True

# ## MLM训练调试参数开始 ## #
gama =0.5
MLMEpochs = 16
WordGenTimes = 10
if WordGenTimes > 1:
    RanWrongDivisor = 1.0
else:
    RanWrongDivisor = 0.15
MLMLearningRate = 1e-4
RepeatNum = 1
BatchSize = 10
SentenceLength = 512
finetunePath = '../checkpoint/finetune/mlm_trained_%s.model' % SentenceLength
PretrainPath = '../checkpoint/pretrain/pytorch_model.bin'
VocabPath = '../checkpoint/pretrain/vocab.txt'

# ## MLM训练调试参数结束 ## #

# ## MLM通用参数 ## #
DropOut = 0.1
MaskRate = 0.15
VocabSize = len(open(VocabPath, 'r', encoding='utf-8').readlines())
HiddenSize = 768
HiddenLayerNum = 12
IntermediateSize = 3072
AttentionHeadNum = 12


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())



# 参数名配对
local2target_emb = {
    'smbert_emd.token_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
    'smbert_emd.type_embeddings.weight': 'bert.embeddings.token_type_embeddings.weight',
    'smbert_emd.position_embeddings.weight': 'bert.embeddings.position_embeddings.weight',
    'smbert_emd.emb_normalization.weight': 'bert.embeddings.LayerNorm.gamma',
    'smbert_emd.emb_normalization.bias': 'bert.embeddings.LayerNorm.beta'
}

local2target_transformer = {
    'transformer_blocks.%s.multi_attention.q_dense.weight': 'bert.encoder.layer.%s.attention.self.query.weight',
    'transformer_blocks.%s.multi_attention.q_dense.bias': 'bert.encoder.layer.%s.attention.self.query.bias',
    'transformer_blocks.%s.multi_attention.k_dense.weight': 'bert.encoder.layer.%s.attention.self.key.weight',
    'transformer_blocks.%s.multi_attention.k_dense.bias': 'bert.encoder.layer.%s.attention.self.key.bias',
    'transformer_blocks.%s.multi_attention.v_dense.weight': 'bert.encoder.layer.%s.attention.self.value.weight',
    'transformer_blocks.%s.multi_attention.v_dense.bias': 'bert.encoder.layer.%s.attention.self.value.bias',
    'transformer_blocks.%s.multi_attention.o_dense.weight': 'bert.encoder.layer.%s.attention.output.dense.weight',
    'transformer_blocks.%s.multi_attention.o_dense.bias': 'bert.encoder.layer.%s.attention.output.dense.bias',
    'transformer_blocks.%s.attention_layernorm.weight': 'bert.encoder.layer.%s.attention.output.LayerNorm.gamma',
    'transformer_blocks.%s.attention_layernorm.bias': 'bert.encoder.layer.%s.attention.output.LayerNorm.beta',
    'transformer_blocks.%s.feedforward.dense1.weight': 'bert.encoder.layer.%s.intermediate.dense.weight',
    'transformer_blocks.%s.feedforward.dense1.bias': 'bert.encoder.layer.%s.intermediate.dense.bias',
    'transformer_blocks.%s.feedforward.dense2.weight': 'bert.encoder.layer.%s.output.dense.weight',
    'transformer_blocks.%s.feedforward.dense2.bias': 'bert.encoder.layer.%s.output.dense.bias',
    'transformer_blocks.%s.feedforward_layernorm.weight': 'bert.encoder.layer.%s.output.LayerNorm.gamma',
    'transformer_blocks.%s.feedforward_layernorm.bias': 'bert.encoder.layer.%s.output.LayerNorm.beta',
}
