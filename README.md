# chinese-t5
chinese-t5-pytorch-generate
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
