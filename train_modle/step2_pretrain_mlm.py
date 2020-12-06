# coding=utf-8
import os,sys,warnings
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.append('Pytorch_SoftMaskBert_SonofGod/')  ##GPU上跑 请设置路径
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from smbert.data.mlm_dataset import *
from smbert.layers.SM_Bert_mlm import SMBertMlm
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    if Debug:
        print('开始训练 %s' % get_time())
    onehot_type = False
    soft_masked_bert = SMBertMlm().to(device)
    if Debug:
        print('Total Parameters:', sum([p.nelement() for p in soft_masked_bert.parameters()]))

    if os.path.exists(PretrainPath):
        print('开始加载预训练模型！')
        soft_masked_bert.load_pretrain(PretrainPath)
        print('完成加载预训练模型！')

    dataset = SMBertDataSet(path_train_save, onehot_type)
    dataloader = DataLoader(dataset=dataset, batch_size=BatchSize, shuffle=True, drop_last=True)
    testset = RobertaTestSet(path_dev_save)

    optim = Adam(soft_masked_bert.parameters(), lr=MLMLearningRate)
    c_criterion = nn.CrossEntropyLoss().to(device)
    d_criterion = nn.BCELoss().to(device)


    for epoch in range(MLMEpochs):
        # train
        soft_masked_bert.train()
        data_iter = tqdm(enumerate(dataloader),
                         desc='EP_%s:%d' % ('train', epoch),
                         total=len(dataloader),
                         bar_format='{l_bar}{r_bar}')
        print_loss = 0.0
        c_correct , d_correct = 0,0
        sen_c_correct , sen_d_correct = 0,0
        char_count , sentence_count = 0,0
        for i, data in data_iter:
            data = {k: v.to(device) for k, v in data.items()}
            input_token = data['input_token_ids']
            segment_ids = data['segment_ids']
            correct_label = data['token_ids_labels']
            label = data['label']

            mlm_output ,prob = soft_masked_bert(input_token, segment_ids)
            mlm_output = mlm_output.permute(0, 2, 1)
            loss_c = c_criterion(mlm_output,correct_label)
            loss_d = d_criterion(prob[:,:,0],label.float())
            mask_loss = gama*loss_c + (1-gama)*loss_d
            ## 训练模型
            print_loss += mask_loss.item()
            optim.zero_grad()
            mask_loss.backward()
            optim.step()

            ## 评价指标
            output_topk = torch.argmax(mlm_output.permute(0,2,1),dim=-1)
            output_prob = torch.round(prob[:,:,0]).long()
            input_token_list = input_token.tolist()
            sentence_count += len(input_token_list)
            for j in range(len(input_token_list)):
                input_len = len([i for i in input_token_list[j] if i]) -2
                char_count += input_len
                ## 字的准确率
                c_correct += torch.sum(torch.eq(correct_label[j][1:input_len+1] , output_topk[j][1:input_len+1])).tolist()
                d_correct += torch.sum(torch.eq(output_prob[j][1:input_len+1] ,label[j][1:input_len+1])).tolist()
                ## 句子准确率
                sen_c_correct += sum([torch.equal( correct_label[j][1:input_len+1] , output_topk[j][1:input_len+1] )])
                sen_d_correct += sum([torch.equal( output_prob[j][1:input_len+1] ,label[j][1:input_len+1] )])

        print('EP_%d mask loss:%s  sen_d_correct:%s  sen_c_correct:%s  char_d_correct:%s  char_c_correct:%s' %
              (epoch, print_loss/(i+1),sen_d_correct/sentence_count ,sen_c_correct/sentence_count,
               d_correct/char_count,c_correct/char_count))

        # save
        output_path = PretrainPath + '.ep%d' % epoch
        torch.save(soft_masked_bert.cpu(), output_path)
        soft_masked_bert.to(device)
        print('EP:%d Model Saved on:%s' % (epoch, output_path))

        # test
        if epoch % 2 ==0:
            with torch.no_grad():
                soft_masked_bert.eval()
                t_sentence_count , t_char_count = 0,0
                t_sen_c_correct,t_sen_d_correct,t_c_correct,t_d_correct = 0,0,0,0
                for test_data in testset:
                    input_token = test_data['input_token_ids'].unsqueeze(0).to(device)
                    segment_ids = test_data['segment_ids'].unsqueeze(0).to(device)
                    input_token_list = input_token.tolist()
                    input_len = len([x for x in input_token_list[0] if x]) - 2
                    label_list = test_data['token_ids_labels'].to(device)
                    label = test_data['label'].to(device)
                    mlm_output,prob = soft_masked_bert(input_token, segment_ids)
                    output_tensor = torch.nn.Softmax(dim=-1)(mlm_output)
                    output_topk = torch.argmax(output_tensor,dim=-1).squeeze(0)
                    output_prob = torch.round(prob[:,:,0]).long().squeeze(0)

                    t_char_count += input_len
                    t_sentence_count += 1

                    ## 字的准确率
                    t_c_correct += torch.sum(torch.eq(correct_label[1:input_len + 1], output_topk[1:input_len + 1])).tolist()
                    t_d_correct += torch.sum(torch.eq(output_prob[1:input_len + 1], label[1:input_len + 1])).tolist()
                    ## 句子准确率
                    t_sen_c_correct += sum([torch.equal(correct_label[1:input_len + 1], output_topk[1:input_len + 1])])
                    t_sen_d_correct += sum([torch.equal(output_prob[1:input_len + 1], label[1:input_len + 1])])
            print('EP_%d 测试集: sen_d_correct:%s  sen_c_correct:%s  char_d_correct:%s  char_c_correct:%s' %
                  (epoch, t_sen_d_correct / sentence_count, t_sen_c_correct / sentence_count,
                   t_d_correct / char_count, t_c_correct / char_count))
