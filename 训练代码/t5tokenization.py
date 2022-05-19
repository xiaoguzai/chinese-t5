import sentencepiece as spm


class T5Tokenizer(object):
    #t5中的inputs和labels千米填充的特殊符号不一样，明天仔细看一下
    def __init__(self,config,vocab_file):
        self.pad_token_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self.decoder_start_token_id = config.decoder_start_token_id
        self.vocab_file = vocab_file
        self.sp_model_kwargs = {}
        self.sp_model = spm.SentencePieceProcessor(self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

    
    def tokenize(self,text):
        tokenized_text = self.sp_model.encode(text,out_type=str)
        return tokenized_text
    
    def get_input_ids(self,text):
        tokens = self.tokenize(text)
        ids = []
        #print('...tokens = ...')
        #print(tokens)
        #print('...............')
        for token in tokens:
            #if token == '▁':
            #    continue
            ids.append(self.sp_model.piece_to_id(token))
        return_text = self.get_input_text(ids)
        #print('...return_text = ...')
        #print(return_text)
        #print('....................')
        ids.append(self.eos_token_id)
        return ids
    
    def _convert_id_to_token(self,index):
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.id_to_piece(index)
        else:
            token = f"<extra_id_{self.vocab_size - 1 - index}>"
        return token
    
    def clean_up_tokenization(self,out_string:str):
        out_string = (
            out_string.replace(" .",".")
            .replace(" 。","。")
            .replace(" ?","?")
            .replace(" !","!")
            .replace(" ,",",")
            .replace(" ' ","'")
            .replace(" n't","n't")
            .replace(" 'm","'m")
            .replace(" 's","'s")
            .replace(" 've","'ve")
            .replace(" 're","'re")
        )
        return out_string
    
    def get_input_text(self,ids):
        text = []
        #for id in ids:
        #    if id in [self.pad_token_id,self.eos_token_id,self.decoder_start_token_id]:
        #        continue
        #    text.append(self.sp_model.id_to_piece(id))
        #text = self.sp_model.id_to_piece(ids)
        for id in ids:
            if id in [self.pad_token_id,self.eos_token_id,self.decoder_start_token_id]:
                continue
            if self._convert_id_to_token(id) == "<unk>":
                continue
            text.append(self._convert_id_to_token(id))
        #阅读官方decode部分内容
        text = "".join(text)
        text = self.clean_up_tokenization(text)
        return text