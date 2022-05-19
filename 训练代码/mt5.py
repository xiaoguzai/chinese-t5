#Author:xiaoguzai
#email:474551240@qq.com
#Download pretrain-file from https://huggingface.co/google/mMT5-base/tree/main
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
r"""
                                                               DecoderLayerAttention
                                       DecoderLayerTransformers
decoder部分的结构图---DecoderTransformers                        
                                                               DecoderLayerAttention
                                       DecoderCrossTransformers
                                                               DecoderCrossAttention
"""
class MT5Config(object):
    def __init__(self,
                 d_ff = 2048,
                 d_kv = 64,
                 d_model = 768,
                 decoder_start_token_id = 0,
                 dropout_rate = 0.1,
                 eos_token_id = 1,
                 feed_forward_proj = 'gated-gelu',
                 initializer_factor = 1.0,
                 is_encoder_decoder = True,
                 layer_norm_epsilon = 1e-06,
                 model_type = 'mMT5',
                 num_decoder_layers = 12,
                 num_heads = 12,
                 num_layers = 12,
                 output_past = True,
                 pad_token_id = 0,
                 relative_attention_num_buckets = 32,
                 tie_word_embeddings = False,
                 tokenizer_class = 'MT5Tokenizer',
                 use_cache = True,
                 vocab_size = 250112,
                *args, **kwargs):
        self.intermediate_size = d_ff
        self.d_ff = d_ff
        
        self.size_per_head = d_kv
        self.embedding_size = d_model

        self.decoder_start_token_id = decoder_start_token_id
        self.dropout_rate = dropout_rate
        self.eos_token_id = eos_token_id
        self.feed_forward_proj = feed_forward_proj
        self.initializer_factor = initializer_factor
        self.is_encoder_decoder = is_encoder_decoder
        self.layer_norm_epsilon = layer_norm_epsilon
        self.model_type = model_type

        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.output_past = output_past
        self.pad_token_id = pad_token_id
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.tie_word_embeddings = tie_word_embeddings
        self.use_cache = use_cache
        self.vocab_size = vocab_size

@torch.no_grad()
def greedy_generate(model,config,input_ids,labels=None,max_length = 20):
    model.eval()
    flag = 0
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    unfinished_sequences = unfinished_sequences[:,None]
    pad_token_id = config.pad_token_id
    eos_token_id = config.eos_token_id
    while True:
        if flag == 0:
            result,layer_key_value_list,cross_key_value_list = model(input_ids,labels=labels)
            result = result[:,-1,:]
            output_id = torch.argmax(result,axis=-1)
            decoder_ids = output_id[:,None]
            output_id = output_id[:,None]*unfinished_sequences + pad_token_id*(1-unfinished_sequences)
            #result_id = torch.cat([input_ids,output_id],dim=-1)
            result_id = output_id
            #这里的input_id的对应值始终保持不变,decoder_ids为不断拼接的结果
            #从而得到最终的result_id的结果
            flag = flag+1
        else:
            result,layer_key_value_list,cross_key_value_list = model(input_ids,labels=decoder_ids,layer_key_value_list=layer_key_value_list,\
                                                                     cross_key_value_list=cross_key_value_list)
            #layer_key_value_list记录上一次整个decoder的过程中产生的key和value的内容
            result = result[:,-1,:]
            output_id = torch.argmax(result,axis=-1)
            decoder_ids = output_id[:,None]
            output_id = output_id[:,None]*unfinished_sequences + pad_token_id*(1-unfinished_sequences)
            result_id = torch.cat([result_id, output_id],dim=-1)
            #这里的input_id的对应值始终保持不变,decoder_ids为不断拼接的结果
            
        flag = flag+1
        unfinished_sequences = unfinished_sequences.mul((decoder_ids != eos_token_id).long())
        if unfinished_sequences.max() == 0 or flag > max_length:
            break
    #print('result_id = ')
    #print(result_id)
    return result_id

class MT5Generation(nn.Module):
    def __init__(self,config,**kwargs):
        super(MT5Generation,self).__init__()
        self.config = config
        self.mt5 = MT5(config)
        self.lm_head = nn.Linear(config.embedding_size,config.vocab_size,bias=False)
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            #一般这里的mean=0.0,stddev标准差为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
     
    def _shift_right(self,input_ids):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.config.decoder_start_token_id
        assert self.config.pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, self.config.pad_token_id)
        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"
        shifted_input_ids.to(input_ids.device)
        return shifted_input_ids
    
    def forward(self,input_ids,labels=None,generate=True,layer_key_value_list=None,cross_key_value_list=None):
        #input_ids.to(device)
        #labels.to(device)
        if generate == False:
            assert (
                labels == None,
            ), f"Train t5 labels cannot be None."
            decoder_ids = self._shift_right(labels)
            decoder_ids.to(labels.device)
        else:
            #生成部分的内容
            if labels == None:
                #第一波生成
                batch_size = input_ids.shape[0]
                decoder_ids = torch.ones((batch_size,1),dtype=torch.long,device=input_ids.device)*self.config.decoder_start_token_id
                decoder_ids.to(input_ids.device)
            else:
                #后续波生成
                decoder_ids = labels
                decoder_ids.to(labels.device)
        self.mt5.to(input_ids.device)
        output_ids,layer_key_value_list,cross_key_value_list = self.mt5(input_ids,labels=decoder_ids,layer_key_value_list=layer_key_value_list,\
                                                                        cross_key_value_list=cross_key_value_list)
        self.lm_head.to(input_ids.device)
        output_ids = self.lm_head(output_ids)
        if generate == False:
            #训练模式
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(output_ids.view(-1,output_ids.size(-1)), labels.view(-1))
            return output_ids,loss
            #too many values to unpack这里代指这个部分
        else:
            return output_ids,layer_key_value_list,cross_key_value_list

class MT5(nn.Module):
    def __init__(self,config,**kwargs):
        #这里初步先将所有的参数都放入__init__之中
        #后期可以将不需要更改的参数放入build函数之中
        #之前看的内容相当于对params更新name，并且将所有属性中带有name的内容进行更新
        #这样就成功定义了name的对应值
        #super(Nezha, self).__init__()
        super(MT5,self).__init__()
        self.config = config
        self.mt5encoder = MT5Encoder(config)
        self.mt5decoder = MT5Decoder(config)
        #可以在MT5模型之中加入贪婪搜索和**搜索
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            #一般这里的mean=0.0,stddev标准差为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def _shift_right(self,input_ids):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.config.decoder_start_token_id
        assert self.config.pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, self.config.pad_token_id)
        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"
        return shifted_input_ids
    
    def forward(self,input_ids,labels=None,layer_key_value_list=None,cross_key_value_list=None):
        decoder_ids = labels
        encoder_attention_mask = input_ids.ne(self.config.pad_token_id).long()
        encoder_attention_mask = (1.0 - encoder_attention_mask)*-10000.0
        encoder_attention_mask = encoder_attention_mask[:,None,None,:]
        encoder_attention_mask = encoder_attention_mask.to(input_ids.device)
        decoder_attention_mask = decoder_ids.ne(self.config.pad_token_id).long()
        decoder_attention_mask = decoder_attention_mask.to(input_ids.device)
        extended_decoder_attention_mask = (1.0 - decoder_attention_mask)*-10000.0
        extended_decoder_attention_mask = extended_decoder_attention_mask.to(input_ids.device)
        encoderoutput,_ = self.mt5encoder(input_ids,encoder_attention_mask)
        output_ids,layer_key_value_list,cross_key_value_list = self.mt5decoder(input_ids=decoder_ids,encoder_output=encoderoutput,\
                                                                              layer_key_value_list=layer_key_value_list,cross_key_value_list=cross_key_value_list)
        return output_ids,layer_key_value_list,cross_key_value_list

def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class MT5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the MT5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states
    
class MT5DenseReluDense(nn.Module):
    #MT5-1.0使用的结构
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.embedding_size, config.intermediate_size, bias=False)
        self.wo = nn.Linear(config.intermediate_size, config.embedding_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = nn.functional.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class MT5DenseGatedGeluDense(nn.Module):
    #MT5-1.1使用的结构(即mMT5使用的结构)
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.embedding_size, config.intermediate_size, bias=False)
        self.wi_1 = nn.Linear(config.embedding_size, config.intermediate_size, bias=False)
        self.wo = nn.Linear(config.intermediate_size, config.embedding_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = gelu_new

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states
    
class MT5Encoder(nn.Module):
    def __init__(self,
                 config,
                 **kwargs):
        super(MT5Encoder,self).__init__()
        self.config = config
        self.mt5encoderembeddings_layer = nn.Embedding(config.vocab_size,config.embedding_size)
        self.mt5embeddingdropout = nn.Dropout(config.dropout_rate)
        self.mt5encoder_layer = nn.ModuleList()
        for _ in range(config.num_layers):
            MT5encoder_layer_attention = MT5EncoderTransformers(config)
            self.mt5encoder_layer.append(MT5encoder_layer_attention)
        self.final_layer_norm = MT5LayerNorm(config.embedding_size,config.layer_norm_epsilon)
        self.final_dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self,input_ids,encoder_attention_mask):
        output = self.mt5encoderembeddings_layer(input_ids)
        output = self.mt5embeddingdropout(output)
        position_bias = None
        past_value_list = None
        for layer_ndx in self.mt5encoder_layer:
            output,position_bias = layer_ndx(output,position_bias,encoder_attention_mask)
        #到这里都一样
        #!!!易错点：这里有一个final_layer_norm网络层以及一个dropout网络层
        output = self.final_layer_norm(output)
        output = self.final_dropout(output)
        return output,position_bias

class MT5Decoder(nn.Module):
    def __init__(self,
                 config,
                 **kwargs):
        super(MT5Decoder,self).__init__()
        self.mt5decoderembeddings_layer = nn.Embedding(config.vocab_size,config.embedding_size)
        self.mt5embeddingdropout = nn.Dropout(config.dropout_rate)
        #decoder调用embedding层进行操作,这里会放入新的输入,所以需要一个embedding网络层
        self.mt5decoderlayer_transformers_list = nn.ModuleList()
        self.mt5decodercross_transformers_list = nn.ModuleList()
        self.config = config
        self.final_layer_norm = MT5LayerNorm(config.embedding_size,config.layer_norm_epsilon)
        self.final_dropout = nn.Dropout(config.dropout_rate)
        past_key_value_list = []
        for index in range(config.num_layers):
            MT5decoderlayer_transformers = MT5DecoderLayerTransformers(config)
            self.mt5decoderlayer_transformers_list.append(MT5decoderlayer_transformers)
            MT5decodercross_transformers = MT5DecoderCrossTransformers(config)
            self.mt5decodercross_transformers_list.append(MT5decodercross_transformers)
        #decoder之中的第一个layerattention并未调用之前encoder的输出结果，
        #到layercrossattention的时候才调用encoder的输出的结果
    def forward(self,input_ids,encoder_output,layer_key_value_list=None,cross_key_value_list=None):
        input_ids = self.mt5decoderembeddings_layer(input_ids)
        input_ids = self.mt5embeddingdropout(input_ids)
        #input_ids = tensor([[250099]]),input_ids = tensor([[[2.1719e+00,-3.9375e+00,...]]])
        #第二波的时候关键是下面的循环调用前面的past_key_value和position_bias的部分
        batch_size,seq_length = input_ids.size()[0],input_ids.size()[1]
        seq_ids = torch.arange(seq_length)
        causal_mask = seq_ids[None,None,:].repeat(batch_size,seq_length,1) <= seq_ids[None,:,None]
        extended_attention_mask = causal_mask[:,None,:,:]
        extended_attention_mask = extended_attention_mask.to(input_ids.dtype)
        extended_attention_mask = (1.0-extended_attention_mask)*(-10000)
        extended_attention_mask = extended_attention_mask.to(input_ids.device)
        if layer_key_value_list == None:
            layer_key_value_list = [None]*self.config.num_layers
        if cross_key_value_list == None:
            cross_key_value_list = [None]*self.config.num_layers
        #layer_key_value_list和cross_key_value_list当前这一波模型计算出来的内容
        layer_position_bias,cross_position_bias = None,None
        for index in range(self.config.num_layers):
            current_decoder_layer_attention = self.mt5decoderlayer_transformers_list[index]
            current_decoder_cross_attention= self.mt5decodercross_transformers_list[index]
            #这里针对layer_attention和cross_attention中的is_first_layer赋值很关键
            #因为前面的is_first_layer值的改变并不会引起后面的is_first_layer的值改变
            #第一次的时候layer_position_bias和cross_position_bias的值都为None
            #后续的时候layer_position_bias和cross_position_bias接着前面的继续使用
            input_ids,past_key_value,layer_position_bias = current_decoder_layer_attention(input_ids,encoder_output,extended_attention_mask,layer_key_value_list[index],layer_position_bias)
            layer_key_value_list[index] = past_key_value
            input_ids,past_key_value,cross_position_bias = current_decoder_cross_attention(input_ids,encoder_output,cross_key_value_list[index],cross_position_bias)
            #第一波计算要单独调用的原因:没有之前的layer_key_value_list[index-1]以及cross_key_value_list[index-1]的past_key_value的信息
            cross_key_value_list[index] = past_key_value
        input_ids = self.final_layer_norm(input_ids)
        input_ids = self.final_dropout(input_ids)
        return input_ids,layer_key_value_list,cross_key_value_list

class MT5DecoderLayerTransformers(nn.Module):
    def __init__(self,
                 config,
                 **kwargs):
        super(MT5DecoderLayerTransformers,self).__init__()
        self.decoderlayerattention = MT5DecoderLayerAttention(config)
        self.layer_norm0 = MT5LayerNorm(config.embedding_size,eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
    def forward(self,input_ids,encoder_output,extended_attention_mask,past_key_value,position_bias):
        origin_input_ids = input_ids
        input_ids = self.layer_norm0(input_ids)
        input_ids,past_key_value,position_bias = self.decoderlayerattention(input_ids,encoder_output,extended_attention_mask,past_key_value,position_bias)
        input_ids = origin_input_ids+self.dropout(input_ids)
        return input_ids,past_key_value,position_bias

class MT5DecoderCrossTransformers(nn.Module):
    def __init__(self,
                 config,
                 **kwargs):
        super(MT5DecoderCrossTransformers,self).__init__()
        self.decodercrossattention = MT5DecoderCrossAttention(config)
        self.layer_norm0 = MT5LayerNorm(config.embedding_size,eps=config.layer_norm_epsilon)
        self.layer_norm1 = MT5LayerNorm(config.embedding_size,eps=config.layer_norm_epsilon)
        self.mt5densegatedgeludense = MT5DenseGatedGeluDense(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = gelu_new
        self.config = config
    def forward(self,input_ids,encoder_output,past_key_value,position_bias):
        origin_input_ids = input_ids
        input_ids = self.layer_norm0(input_ids)
        input_ids,past_key_value,position_bias = self.decodercrossattention(input_ids,encoder_output,past_key_value,position_bias)
        input_ids = origin_input_ids+self.dropout(input_ids)
        if torch.isinf(input_ids).any():
            clamp_value = torch.finfo(input_ids.dtype).max - 1000
            input_ids = torch.clamp(input_ids, min=-clamp_value, max=clamp_value)
        origin_input_ids = input_ids
        input_ids = self.layer_norm1(input_ids)
        input_ids = self.mt5densegatedgeludense(input_ids)
        input_ids = origin_input_ids+self.dropout(input_ids)
        return input_ids,past_key_value,position_bias

class MT5EncoderTransformers(nn.Module):
#这里目前选用的是MT5-1.1的版本，效果更好
    def __init__(self,
                 config,
                 **kwargs):
        super(MT5EncoderTransformers,self).__init__()
        self.mt5encoderlayerattention = MT5EncoderLayerAttention(config)
        self.mt5layernorm0 = MT5LayerNorm(config.embedding_size,config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        #这里注意使用nn.Dropout,F.Dropout之中有坑
        self.mt5densegatedgeludense = MT5DenseGatedGeluDense(config)
        self.mt5layernorm1 = MT5LayerNorm(config.embedding_size,config.layer_norm_epsilon)
    
    def forward(self,input_ids,position_bias,encoder_attention_mask):
        origin_input_ids = input_ids
        input_ids = self.mt5layernorm0(input_ids)
        input_ids,position_bias = self.mt5encoderlayerattention(input_ids,position_bias,encoder_attention_mask)
        input_ids = origin_input_ids+self.dropout(input_ids)
        #到这里也是一样的
        origin_input_ids = input_ids
        input_ids = self.mt5layernorm1(input_ids)
        input_ids = self.mt5densegatedgeludense(input_ids)
        input_ids = origin_input_ids+self.dropout(input_ids)
        #还是这里的self.dropout出问题!!!
        return input_ids,position_bias

class MT5EncoderLayerAttention(nn.Module):
    def __init__(self,
                 config,
                 **kwargs):
        super(MT5EncoderLayerAttention,self).__init__()
        self.query_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        #这里MT5-base的时候config.num_heads*config.size_per_head == config.embedding_size,
        #MT5-small的时候两者并不相等
        self.key_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        self.value_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        self.output_layer = nn.Linear(config.num_heads*config.size_per_head,config.embedding_size,bias=False)
        self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets,config.num_heads)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.config = config
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            #一般这里的mean=0.0,stddev标准差为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        (我们对于小的相对绝对位置使用小的bucket，对于大的相对绝对位置使用大的buckets，所有相对位置超出max_distance都映射到相同的bucket
        之中，所有相对位置小于等于max_distance都映射到相同的bucket之中，这比模型训练允许更多更优雅的更长序列的归一化操作)
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=True,
            num_buckets=self.config.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        #这里的relative_attention_bias调用nn.embedding的模型内容
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    
    def forward(self,input_ids,position_bias=None,encoder_attention_mask=None):
        #position_bias通过传递的方式，减少模型的计算过程，加快模型的运算速度
        batch_size,seq_length = input_ids.shape[:2]
        query = self.query_layer(input_ids)
        key = self.key_layer(input_ids)
        value = self.value_layer(input_ids)
        query = query.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        #-1的意思是这个位置让电脑帮我们计算具体的维度内容
        key = key.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        value = value.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        
        scores = torch.matmul(
            query,key.transpose(3,2)
        )
        real_seq_length,key_length = input_ids.shape[1],input_ids.shape[1]
        if position_bias == None:
            position_bias = self.compute_bias(real_seq_length,key_length)
        if encoder_attention_mask != None:
            position_bias = position_bias+encoder_attention_mask
        scores += position_bias
        #到这里目前内容一致
        attn_weights = F.softmax(scores,dim=-1)
        #!!!这里的softmax一定要注明dim=-1
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights,value)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size,-1,self.config.num_heads*self.config.size_per_head)
        attn_output = self.output_layer(attn_output)
        #the same
        return attn_output,position_bias

class MT5DecoderLayerAttention(nn.Module):
    def __init__(self,
                 config,
                 **kwargs):
        super(MT5DecoderLayerAttention,self).__init__()
        self.query_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        #这里MT5-base的时候config.num_heads*config.size_per_head == config.embedding_size,
        #MT5-small的时候两者并不相等
        self.key_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        self.value_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        self.output_layer = nn.Linear(config.num_heads*config.size_per_head,config.embedding_size,bias=False)
        self.relative_attention_bias = nn.Embedding(config.relative_attention_num_buckets,config.num_heads)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.config = config
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            #一般这里的mean=0.0,stddev标准差为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=False, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        (我们对于小的相对绝对位置使用小的bucket，对于大的相对绝对位置使用大的buckets，所有相对位置超出max_distance都映射到相同的bucket
        之中，所有相对位置小于等于max_distance都映射到相同的bucket之中，这比模型训练允许更多更优雅的更长序列的归一化操作)
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=False,
            num_buckets=self.config.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        #这里的relative_attention_bias调用nn.embedding的模型内容
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    
    def forward(self,input_ids,encoder_output,extended_attention_mask,past_key_value=None,position_bias=None):
        #position_bias通过传递的方式，减少模型的计算过程，加快模型的运算速度
        batch_size,seq_length = input_ids.shape[:2]
        real_seq_length,key_length = input_ids.shape[1],input_ids.shape[1]
        if past_key_value is not None:
            #decoder第二个单词之后的内容时
            real_seq_length += past_key_value[0].shape[2]
            key_length += past_key_value[0].shape[2]
        query = self.query_layer(input_ids)
        key = self.key_layer(input_ids)
        value = self.value_layer(input_ids)
        
        query = query.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        #-1的意思是这个位置让电脑帮我们计算具体的维度内容
        key = key.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        value = value.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        if past_key_value != None:
            key = torch.cat([past_key_value[0],key],dim=2)
        if past_key_value != None:
            value = torch.cat([past_key_value[1],value],dim=2)
        scores = torch.matmul(
            query,key.transpose(3,2)
        )
        past_key_value = [key,value]
        #!!!注意这里的key所在的位置
        if position_bias == None:
            #在decoderlayerselfattention中使用的是计算bias的函数内容
            position_bias = self.compute_bias(real_seq_length,key_length)
            #position_bias.shape = (1,12,2,2),scores.shape = (2,12,1,2)
        if past_key_value is not None:
            position_bias = position_bias[:, :, -input_ids.size(1) :, :]
            #position_bias.shape = (1,12,1,2)，将多出来的部分去除掉
        #position_bias = position_bias.to(input_ids.device)
        position_bias = position_bias+extended_attention_mask
        #position_bias = position_bias.to(input_ids.device)

        scores += position_bias
        attn_weights = F.softmax(scores,dim=-1)
        #这里必须加上dim=-1,默认的dim应该等于1
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights,value)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size,-1,self.config.num_heads*self.config.size_per_head)
        attn_output = self.output_layer(attn_output)
        return attn_output,past_key_value,position_bias

class MT5DecoderCrossAttention(nn.Module):
    def __init__(self,
                 config,
                 **kwargs):
        super(MT5DecoderCrossAttention,self).__init__()
        self.query_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        #这里MT5-base的时候config.num_heads*config.size_per_head == config.embedding_size,
        #MT5-small的时候两者并不相等
        self.key_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        self.value_layer = nn.Linear(config.embedding_size,config.num_heads*config.size_per_head,bias=False)
        self.output_layer = nn.Linear(config.num_heads*config.size_per_head,config.embedding_size,bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.config = config
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            #一般这里的mean=0.0,stddev标准差为1
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=False, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        (我们对于小的相对绝对位置使用小的bucket，对于大的相对绝对位置使用大的buckets，所有相对位置超出max_distance都映射到相同的bucket
        之中，所有相对位置小于等于max_distance都映射到相同的bucket之中，这比模型训练允许更多更优雅的更长序列的归一化操作)
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(
            query_length, dtype=torch.long, device=self.query_layer.weight.device
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device=self.query_layer.weight.device
        )[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=False,
            num_buckets=self.config.relative_attention_num_buckets,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        #这里的relative_attention_bias调用nn.embedding的模型内容
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    
    def forward(self,input_ids,encoder_output,past_key_value=None,position_bias=None):
        #position_bias通过传递的方式，减少模型的计算过程，加快模型的运算速度
        batch_size,seq_length = input_ids.shape[:2]
        real_seq_length,key_length = input_ids.shape[1],input_ids.shape[1]
        if past_key_value is not None:
            #decoder第二个单词之后的内容时
            real_seq_length += past_key_value[0].shape[2]
            key_length += past_key_value[0].shape[2]
        if encoder_output is not None:
            key_length = encoder_output.shape[1]
        query = self.query_layer(input_ids)
        query = query.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        #这里input_ids与past_key_value[0]连接之前需要变换维度
        key = self.key_layer(encoder_output)
        #past_key_value[0]保存上一波的key的值        
        if past_key_value != None:
            key = past_key_value[0]
        else:
            key = key.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        #key.size = (2,5,768)
        value = self.value_layer(encoder_output)
        if past_key_value != None:
            #value = torch.cat([past_key_value[1],value],dim=2)
            value = past_key_value[1]
        else:
            value = value.view(batch_size,-1,self.config.num_heads,self.config.size_per_head).transpose(1,2)
        scores = torch.matmul(
            query,key.transpose(3,2)
        )
        if position_bias is None:
            #在decodercrossattention中使用的为全零的矩阵
            encoder_length = encoder_output.shape[1]
            position_bias = torch.zeros(1,self.config.num_heads,seq_length,encoder_length)
        position_bias = position_bias.to(input_ids.device)
        past_key_value = [key,value]
        scores += position_bias
        attn_weights = F.softmax(scores,dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights,value)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size,-1,self.config.num_heads*self.config.size_per_head)
        attn_output = self.output_layer(attn_output)
        return attn_output,past_key_value,position_bias