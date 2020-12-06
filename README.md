# 复现：Soft-Masked-Bert
- 更清晰的torch版Soft-Masked-Bert
论模型的效果和便利，我还是站bert。
知乎专栏和CSDN专栏会有详细的解读。

## 错别字纠错的模型使用
- 第一步，下载pytorch版的bert预训练的模型，放入checkpoint/pretrain中。
- 第二步，将train、test预料，分别放入data/train_data、data/test_data中。
- 第三步，进入train_modle，运行stp1_gen_train_test.py生成对应的训练和测试集。
- 第四步，打开根目录的pretrain_config.py设置你需要的参数。
- 第五步，修改好参数后，即可运行python3 step2_pretrain_mlm.py来训练了，训练生成的模型保存在checkpoint/finetune里。
- 第五步，如果你需要预测并测试你的模型，则需要运行根目录下的step3_inference.py。需要注意的事，你需要将训练生成的模型改名成：mlm_trained_xx.model。

