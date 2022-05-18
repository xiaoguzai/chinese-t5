# chinese-t5

基于追一科技发布的t5模型在pytorch上面的调用

[官方链接](https://github.com/ZhuiyiTechnology/t5-pegasus)

感谢大佬转化了一下权重，可以在transformers库中直接适配

[转换链接](https://github.com/renmada/t5-pegasus-pytorch)

几个对应的参数链接：

t5-copy-summary的对应链接[https://huggingface.co/imxly/t5-copy-summary]

t5-copy的对应链接[https://huggingface.co/imxly/t5-copy]

t5-pegasus的对应链接[https://huggingface.co/imxly/t5-pegasus]

t5-pegasus-small的对应链接[https://huggingface.co/imxly/t5-pegasus-small-]

由于之前的转化链接中的T5PegasusTokenizer分词方法在升级之后的Transformers库之中已经无法找到了，而追一的t5本质上就是调用BertTokenizer分词，所以这里直接替换就好了

```python
#from tokenizer import T5PegasusTokenizer
#from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from transformers import BertTokenizer
from transformers import MT5ForConditionalGeneration

model_path = '/home/xiaoguzai/模型/t5-copy'
model = MT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
text = '蓝蓝的天上有一朵白白的云'
ids = tokenizer.encode(text, return_tensors='pt')
output = model.generate(ids,
                        decoder_start_token_id=tokenizer.cls_token_id,
                        eos_token_id=tokenizer.sep_token_id,
                        max_length=30).numpy()[0]
print(''.join(tokenizer.decode(output[1:])).replace(' ', ''))
```



```
原始生成结果：蓝蓝的天上有一朵白白的云。蓝蓝的天上有一朵白白的云。蓝蓝的
```

ps:这里的tokenizer.cls_token_id = 101,tokenizer.sep_token_id = 102,因此这里如果修改config.json中的参数,

```
decoder_start_token_id = tokenizer.cls_token_id = 101,
eos_token_id = tokenizer.sep_token_id = 102
```

这种情况下就可以在generate函数之中去除掉

```python
decoder_start_token_id = tokenizer.cls_token_id,
eos_token_id = tokenizer.sep_token_id
```

这几个参数

## 原始t5模型在中文文本上存在的问题

原始的t5使用的是sentencepiecemodel切分词语，这一切词方法最大的问题在于中文切词部分非常的不准确，并且它老是以'_'作为开头的位置

每次以下划线进行打头的情况下很影响文本内容的生成，比如我调用一下t5的切分词语部分进行分词

```python
import sentencepiece as spm
sp_model = spm.SentencePieceProcessor("/home/xiaoguzai/模型/t5-base/spiece.model")
token_result = sp_model.encode("我是一个小小的菜鸡。",out_type=str)
print('token_result = ')
print(token_result)
ids = []
for token in token_result:
    ids.append(sp_model.piece_to_id(token))
print('ids = ')
print(ids)
```

这里切词语部分的结果

```
token_result = ['_','我是一个小小的菜鸡。']
ids = [3,2]
```

可以看出原始文本的分词非常的不准
