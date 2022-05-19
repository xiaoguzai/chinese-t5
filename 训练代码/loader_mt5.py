import torch
import copy

def load_mt5_generation_data(model,resolved_archive_file):
    state_dict = None
    if state_dict is None:
        try:
            state_dict = torch.load(resolved_archive_file, map_location="cpu")
            file_name = list(state_dict.keys())
            #这里修改ordered_dict加入新的内容,如果有gamma,beta的情况转换为
            #weight,bias的情况
            r"""
            print('file_name = ')
            print(file_name)
            print('============')
            
            print('state_dict = ')
            print(state_dict)
            print('=============')
            print('shared_weight = ')
            print(state_dict['shared_weight'])
            print('================')
            
            print('state_dict lm_head.weight = ')
            print(state_dict['lm_head.weight'])
            print('============================')
            """
            for name in file_name:
                origin_name = name
                name_list = name.split('.')
                if name_list[-1] == 'gamma':
                    name_list[-1] = 'weight'
                elif name_list[-1] == 'beta':
                    name_list[-1] = 'bias'
                new_name = '.'.join(name_list)
                if new_name != origin_name:
                    state_dict[new_name] = copy.deepcopy(state_dict[origin_name])
                    del state_dict[origin_name]

            #print('!!!file_name = !!!')
            #print(file_name)
            #print('!!!!!!!!!!!!!!!!!!')
            #后面是根据file_name来寻找的，所以一定要重新设定file_name
            model_dict = model.state_dict()
        except Exception:
            raise OSError(
                f"Unable to load weights from pytorch checkpoint file for bert"
                f"at '{resolved_archive_file}'"
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
            )
        transformer_dicts = {
           'mt5.mt5encoder.mt5encoderembeddings_layer.weight':'encoder.embed_tokens.weight',
           'mt5.mt5decoder.mt5decoderembeddings_layer.weight':'decoder.embed_tokens.weight',
           #'mt5.mt5encoder.mt5encoderembeddings_layer.weight':'shared.weight',
           #'mt5.mt5decoder.mt5decoderembeddings_layer.weight':'shared.weight',
           'mt5.mt5encoder.final_layer_norm.weight':'encoder.final_layer_norm.weight',
           'mt5.mt5decoder.final_layer_norm.weight':'decoder.final_layer_norm.weight',
           'lm_head.weight':'lm_head.weight'
        }
        #开头有个shared_weight暂时没用
        #由自己的权重名称去找原先的权重名称
        for layer_ndx in range(model.config.num_layers):
            transformer_dicts.update({
                'mt5.mt5encoder.mt5encoder_layer.%d.mt5encoderlayerattention.query_layer.weight'%(layer_ndx):'encoder.block.%d.layer.0.SelfAttention.q.weight'%(layer_ndx),
                #注意中间有冒号，两边要分开进行赋值
                'mt5.mt5encoder.mt5encoder_layer.%d.mt5encoderlayerattention.key_layer.weight'%(layer_ndx):'encoder.block.%d.layer.0.SelfAttention.k.weight'%(layer_ndx),
                'mt5.mt5encoder.mt5encoder_layer.%d.mt5encoderlayerattention.value_layer.weight'%(layer_ndx):'encoder.block.%d.layer.0.SelfAttention.v.weight'%(layer_ndx),
                'mt5.mt5encoder.mt5encoder_layer.%d.mt5encoderlayerattention.output_layer.weight'%(layer_ndx):'encoder.block.%d.layer.0.SelfAttention.o.weight'%(layer_ndx),
                'mt5.mt5encoder.mt5encoder_layer.%d.mt5encoderlayerattention.relative_attention_bias.weight'%(layer_ndx):'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
                
                'mt5.mt5encoder.mt5encoder_layer.%d.mt5layernorm0.weight'%(layer_ndx):'encoder.block.%d.layer.0.layer_norm.weight'%(layer_ndx),
                'mt5.mt5encoder.mt5encoder_layer.%d.mt5densegatedgeludense.wi_0.weight'%(layer_ndx):'encoder.block.%d.layer.1.DenseReluDense.wi_0.weight'%(layer_ndx),
                'mt5.mt5encoder.mt5encoder_layer.%d.mt5densegatedgeludense.wi_1.weight'%(layer_ndx):'encoder.block.%d.layer.1.DenseReluDense.wi_1.weight'%(layer_ndx),
                'mt5.mt5encoder.mt5encoder_layer.%d.mt5densegatedgeludense.wo.weight'%(layer_ndx):'encoder.block.%d.layer.1.DenseReluDense.wo.weight'%(layer_ndx),
                'mt5.mt5encoder.mt5encoder_layer.%d.mt5layernorm1.weight'%(layer_ndx):'encoder.block.%d.layer.1.layer_norm.weight'%(layer_ndx),
                
                
                'mt5.mt5decoder.mt5decoderlayer_transformers_list.%d.decoderlayerattention.query_layer.weight'%(layer_ndx):'decoder.block.%d.layer.0.SelfAttention.q.weight'%(layer_ndx),
                'mt5.mt5decoder.mt5decoderlayer_transformers_list.%d.decoderlayerattention.key_layer.weight'%(layer_ndx):'decoder.block.%d.layer.0.SelfAttention.k.weight'%(layer_ndx),
                'mt5.mt5decoder.mt5decoderlayer_transformers_list.%d.decoderlayerattention.value_layer.weight'%(layer_ndx):'decoder.block.%d.layer.0.SelfAttention.v.weight'%(layer_ndx),
                'mt5.mt5decoder.mt5decoderlayer_transformers_list.%d.decoderlayerattention.output_layer.weight'%(layer_ndx):'decoder.block.%d.layer.0.SelfAttention.o.weight'%(layer_ndx),
                'mt5.mt5decoder.mt5decoderlayer_transformers_list.%d.decoderlayerattention.relative_attention_bias.weight'%(layer_ndx):'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
                
                'mt5.mt5decoder.mt5decoderlayer_transformers_list.%d.layer_norm0.weight'%(layer_ndx):'decoder.block.%d.layer.0.layer_norm.weight'%(layer_ndx),
                
                'mt5.mt5decoder.mt5decodercross_transformers_list.%d.decodercrossattention.query_layer.weight'%(layer_ndx):'decoder.block.%d.layer.1.EncDecAttention.q.weight'%(layer_ndx),
                'mt5.mt5decoder.mt5decodercross_transformers_list.%d.decodercrossattention.key_layer.weight'%(layer_ndx):'decoder.block.%d.layer.1.EncDecAttention.k.weight'%(layer_ndx),
                'mt5.mt5decoder.mt5decodercross_transformers_list.%d.decodercrossattention.value_layer.weight'%(layer_ndx):'decoder.block.%d.layer.1.EncDecAttention.v.weight'%(layer_ndx),
                'mt5.mt5decoder.mt5decodercross_transformers_list.%d.decodercrossattention.output_layer.weight'%(layer_ndx):'decoder.block.%d.layer.1.EncDecAttention.o.weight'%(layer_ndx),
                
                'mt5.mt5decoder.mt5decodercross_transformers_list.%d.layer_norm0.weight'%(layer_ndx):'decoder.block.%d.layer.1.layer_norm.weight'%(layer_ndx), 
                'mt5.mt5decoder.mt5decodercross_transformers_list.%d.layer_norm1.weight'%(layer_ndx):'decoder.block.%d.layer.2.layer_norm.weight'%(layer_ndx),
                
                'mt5.mt5decoder.mt5decodercross_transformers_list.%d.mt5densegatedgeludense.wi_0.weight'%(layer_ndx):'decoder.block.%d.layer.2.DenseReluDense.wi_0.weight'%(layer_ndx), 
                'mt5.mt5decoder.mt5decodercross_transformers_list.%d.mt5densegatedgeludense.wi_1.weight'%(layer_ndx):'decoder.block.%d.layer.2.DenseReluDense.wi_1.weight'%(layer_ndx),
                'mt5.mt5decoder.mt5decodercross_transformers_list.%d.mt5densegatedgeludense.wo.weight'%(layer_ndx):'decoder.block.%d.layer.2.DenseReluDense.wo.weight'%(layer_ndx),})
                
                #'mt5decoder.mt5decodercross_transformers_list.%d.relative_attention_bias.weight':'shared.weight'})
                #'mt5decoder.mt5decodercross_transformers_list.%d.decodercrossattention.relative_attention_bias.weight':'shared.weight'})
        model_name = model.state_dict().keys()
        r"""
        print('model_name = ')
        print(model_name)
        print('=============')
        """
        weight_value_tuples = []
        skipped_weight_value_tuples = []
        skip_count = 0
        loaded_weights = []
        used_name = []
        for param_name in model_name:
            stock_name = transformer_dicts[param_name]
            if stock_name in file_name:
                stock_value = state_dict[stock_name]
                param_value = model_dict[param_name]
                if stock_name == 'bert.embeddings.word_embeddings.weight':
                    stock_value = stock_value[:param_value.shape[0]]
                if param_name == 'mlm_dense1.bias':
                    stock_value = stock_value[:param_value.shape[0]]
                if param_name == 'mlm_dense1.weight':
                     stock_value = stock_value.permute(0,1)
                if param_value.shape != stock_value.shape:
                    print("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                          "with the checkpoint:[{}] shape:{}".format(param_name, param_value.shape,
                                                                 stock_name, stock_value.shape))
                    skipped_weight_value_tuples.append((param_name,stock_value))
                    continue
                used_name.append(stock_name)
                model_dict[param_name] = stock_value
                weight_value_tuples.append((param_value,stock_value))
            else:
                print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param_name, stock_name, resolved_archive_file))
                skip_count += 1

    model.load_state_dict(model_dict)
    print("Done loading {} mt5 weights from: {}. "
          "Count of weights not found in the checkpoint was: [{}]. "
          "Count of weights with mismatched shape: [{}]".format(
              len(weight_value_tuples), resolved_archive_file,skip_count, len(skipped_weight_value_tuples)))

    #print("Unused weights from checkpoint:",
    #      "\n\t" + "\n\t".join(sorted(file_name.difference(used_name))))
    print("Unused weights from checkpoint:",
          "\n\t" + "\n\t".join(set(file_name).difference(set(used_name))))
    #stock_weights为从bert之中读取出来的参数矩阵，而loaded_weights为
    #从权重矩阵中加载出来的矩阵，
    return model

def load_mt5_model_data(model,resolved_archive_file):
    state_dict = None
    if state_dict is None:
        try:
            state_dict = torch.load(resolved_archive_file, map_location="cpu")
            file_name = list(state_dict.keys())
            #这里修改ordered_dict加入新的内容,如果有gamma,beta的情况转换为
            #weight,bias的情况
            r"""
            print('file_name = ')
            print(file_name)
            print('============')
            print('state_dict = ')
            print(state_dict)
            print('=============')
            print('shared_weight = ')
            print(state_dict['shared_weight'])
            print('================')
            """
            print('state_dict lm_head.weight = ')
            print(state_dict['lm_head.weight'])
            print('============================')
            
            for name in file_name:
                origin_name = name
                name_list = name.split('.')
                if name_list[-1] == 'gamma':
                    name_list[-1] = 'weight'
                elif name_list[-1] == 'beta':
                    name_list[-1] = 'bias'
                new_name = '.'.join(name_list)
                if new_name != origin_name:
                    state_dict[new_name] = copy.deepcopy(state_dict[origin_name])
                    del state_dict[origin_name]

            #print('!!!file_name = !!!')
            #print(file_name)
            #print('!!!!!!!!!!!!!!!!!!')
            #后面是根据file_name来寻找的，所以一定要重新设定file_name
            model_dict = model.state_dict()
        except Exception:
            raise OSError(
                f"Unable to load weights from pytorch checkpoint file for bert"
                f"at '{resolved_archive_file}'"
                "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
            )
        transformer_dicts = {
           'mt5encoder.mt5encoderembeddings_layer.weight':'encoder.embed_tokens.weight',
           'mt5decoder.mt5decoderembeddings_layer.weight':'decoder.embed_tokens.weight',
           'mt5encoder.final_layer_norm.weight':'encoder.final_layer_norm.weight',
           'mt5decoder.final_layer_norm.weight':'decoder.final_layer_norm.weight',
        }
        #开头有个shared_weight暂时没用
        #由自己的权重名称去找原先的权重名称
        for layer_ndx in range(model.config.num_layers):
            transformer_dicts.update({
                'mt5encoder.mt5encoder_layer.%d.mt5encoderlayerattention.query_layer.weight'%(layer_ndx):'encoder.block.%d.layer.0.SelfAttention.q.weight'%(layer_ndx),
                #注意中间有冒号，两边要分开进行赋值
                'mt5encoder.mt5encoder_layer.%d.mt5encoderlayerattention.key_layer.weight'%(layer_ndx):'encoder.block.%d.layer.0.SelfAttention.k.weight'%(layer_ndx),
                'mt5encoder.mt5encoder_layer.%d.mt5encoderlayerattention.value_layer.weight'%(layer_ndx):'encoder.block.%d.layer.0.SelfAttention.v.weight'%(layer_ndx),
                'mt5encoder.mt5encoder_layer.%d.mt5encoderlayerattention.output_layer.weight'%(layer_ndx):'encoder.block.%d.layer.0.SelfAttention.o.weight'%(layer_ndx),
                'mt5encoder.mt5encoder_layer.%d.mt5encoderlayerattention.relative_attention_bias.weight'%(layer_ndx):'encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
                
                'mt5encoder.mt5encoder_layer.%d.mt5layernorm0.weight'%(layer_ndx):'encoder.block.%d.layer.0.layer_norm.weight'%(layer_ndx),
                'mt5encoder.mt5encoder_layer.%d.mt5densegatedgeludense.wi_0.weight'%(layer_ndx):'encoder.block.%d.layer.1.DenseReluDense.wi_0.weight'%(layer_ndx),
                'mt5encoder.mt5encoder_layer.%d.mt5densegatedgeludense.wi_1.weight'%(layer_ndx):'encoder.block.%d.layer.1.DenseReluDense.wi_1.weight'%(layer_ndx),
                'mt5encoder.mt5encoder_layer.%d.mt5densegatedgeludense.wo.weight'%(layer_ndx):'encoder.block.%d.layer.1.DenseReluDense.wo.weight'%(layer_ndx),
                'mt5encoder.mt5encoder_layer.%d.mt5layernorm1.weight'%(layer_ndx):'encoder.block.%d.layer.1.layer_norm.weight'%(layer_ndx),
                
                
                'mt5decoder.mt5decoderlayer_transformers_list.%d.decoderlayerattention.query_layer.weight'%(layer_ndx):'decoder.block.%d.layer.0.SelfAttention.q.weight'%(layer_ndx),
                'mt5decoder.mt5decoderlayer_transformers_list.%d.decoderlayerattention.key_layer.weight'%(layer_ndx):'decoder.block.%d.layer.0.SelfAttention.k.weight'%(layer_ndx),
                'mt5decoder.mt5decoderlayer_transformers_list.%d.decoderlayerattention.value_layer.weight'%(layer_ndx):'decoder.block.%d.layer.0.SelfAttention.v.weight'%(layer_ndx),
                'mt5decoder.mt5decoderlayer_transformers_list.%d.decoderlayerattention.output_layer.weight'%(layer_ndx):'decoder.block.%d.layer.0.SelfAttention.o.weight'%(layer_ndx),
                'mt5decoder.mt5decoderlayer_transformers_list.%d.decoderlayerattention.relative_attention_bias.weight'%(layer_ndx):'decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight',
                
                'mt5decoder.mt5decoderlayer_transformers_list.%d.layer_norm0.weight'%(layer_ndx):'decoder.block.%d.layer.0.layer_norm.weight'%(layer_ndx),
                
                'mt5decoder.mt5decodercross_transformers_list.%d.decodercrossattention.query_layer.weight'%(layer_ndx):'decoder.block.%d.layer.1.EncDecAttention.q.weight'%(layer_ndx),
                'mt5decoder.mt5decodercross_transformers_list.%d.decodercrossattention.key_layer.weight'%(layer_ndx):'decoder.block.%d.layer.1.EncDecAttention.k.weight'%(layer_ndx),
                'mt5decoder.mt5decodercross_transformers_list.%d.decodercrossattention.value_layer.weight'%(layer_ndx):'decoder.block.%d.layer.1.EncDecAttention.v.weight'%(layer_ndx),
                'mt5decoder.mt5decodercross_transformers_list.%d.decodercrossattention.output_layer.weight'%(layer_ndx):'decoder.block.%d.layer.1.EncDecAttention.o.weight'%(layer_ndx),
                
                'mt5decoder.mt5decodercross_transformers_list.%d.layer_norm0.weight'%(layer_ndx):'decoder.block.%d.layer.1.layer_norm.weight'%(layer_ndx), 
                'mt5decoder.mt5decodercross_transformers_list.%d.layer_norm1.weight'%(layer_ndx):'decoder.block.%d.layer.2.layer_norm.weight'%(layer_ndx),
                
                'mt5decoder.mt5decodercross_transformers_list.%d.mt5densegatedgeludense.wi_0.weight'%(layer_ndx):'decoder.block.%d.layer.2.DenseReluDense.wi_0.weight'%(layer_ndx), 
                'mt5decoder.mt5decodercross_transformers_list.%d.mt5densegatedgeludense.wi_1.weight'%(layer_ndx):'decoder.block.%d.layer.2.DenseReluDense.wi_1.weight'%(layer_ndx),
                'mt5decoder.mt5decodercross_transformers_list.%d.mt5densegatedgeludense.wo.weight'%(layer_ndx):'decoder.block.%d.layer.2.DenseReluDense.wo.weight'%(layer_ndx),})
                
                #'mt5decoder.mt5decodercross_transformers_list.%d.relative_attention_bias.weight':'shared.weight'})
                #'mt5decoder.mt5decodercross_transformers_list.%d.decodercrossattention.relative_attention_bias.weight':'shared.weight'})
        model_name = model.state_dict().keys()
        r"""
        print('model_name = ')
        print(model_name)
        print('=============')
        """
        weight_value_tuples = []
        skipped_weight_value_tuples = []
        skip_count = 0
        loaded_weights = []
        used_name = []
        for param_name in model_name:
            stock_name = transformer_dicts[param_name]
            if stock_name in file_name:
                stock_value = state_dict[stock_name]
                param_value = model_dict[param_name]
                if stock_name == 'bert.embeddings.word_embeddings.weight':
                    stock_value = stock_value[:param_value.shape[0]]
                if param_name == 'mlm_dense1.bias':
                    stock_value = stock_value[:param_value.shape[0]]
                if param_name == 'mlm_dense1.weight':
                     stock_value = stock_value.permute(0,1)
                if param_value.shape != stock_value.shape:
                    print("loader: Skipping weight:[{}] as the weight shape:[{}] is not compatible "
                          "with the checkpoint:[{}] shape:{}".format(param_name, param_value.shape,
                                                                 stock_name, stock_value.shape))
                    skipped_weight_value_tuples.append((param_name,stock_value))
                    continue
                used_name.append(stock_name)
                model_dict[param_name] = stock_value
                weight_value_tuples.append((param_value,stock_value))
            else:
                print("loader: No value for:[{}], i.e.:[{}] in:[{}]".format(param_name, stock_name, resolved_archive_file))
                skip_count += 1

    model.load_state_dict(model_dict)
    print("Done loading {} mt5 weights from: {}. "
          "Count of weights not found in the checkpoint was: [{}]. "
          "Count of weights with mismatched shape: [{}]".format(
              len(weight_value_tuples), resolved_archive_file,skip_count, len(skipped_weight_value_tuples)))

    #print("Unused weights from checkpoint:",
    #      "\n\t" + "\n\t".join(sorted(file_name.difference(used_name))))
    print("Unused weights from checkpoint:",
          "\n\t" + "\n\t".join(set(file_name).difference(set(used_name))))
    #stock_weights为从bert之中读取出来的参数矩阵，而loaded_weights为
    #从权重矩阵中加载出来的矩阵，
    return model

def _load_state_dict_into_model(model,state_dict,pretrained_model_name_or_path,_fast_init=True):
    new_state_dict = model.state_dict()
    print('new_state_dict = ')
    print(new_state_dict)
    return None,None,None,None