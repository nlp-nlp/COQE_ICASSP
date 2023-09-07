import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention, BertSelfAttention
from pdb import set_trace as stop

class SetDecoder_sub_absa(nn.Module):
    def __init__(self, config, num_generated_triples, num_layers, num_classes, return_intermediate=False):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_generated_triples = num_generated_triples
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.query_embed = nn.Embedding(num_generated_triples, config.hidden_size)
        # self.decoder2class = nn.Linear(config.hidden_size, num_classes)
        self.decoder2compare_class = nn.Linear(config.hidden_size, num_classes) # <sub,obj,asp>时候，是做二分类
        self.decoder2span = nn.Linear(config.hidden_size, 4)

        self.metric_1 = nn.Linear(config.hidden_size, config.hidden_size * 8)
        self.metric_2 = nn.Linear(config.hidden_size, config.hidden_size * 8)
        self.metric_3 = nn.Linear(config.hidden_size * 8, 1 * 8, bias=False)

        torch.nn.init.orthogonal_(self.metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.metric_3.weight, gain=1)

        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)


    def forward(self, encoder_hidden_states, encoder_attention_mask):
        """
        encoder_hidden_states: [bsz, enc_len, hidden]
        encoder_attention_mask: [bsz, enc_len, enc_len]
        """
        bsz = encoder_hidden_states.size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        # bsz, q_num, hidden
        hidden_states = self.dropout(self.LayerNorm(hidden_states))
        all_hidden_states = ()
        for i, layer_module in enumerate(self.layers):
            if self.return_intermediate:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

        # bsz, q_num, hidden
        # class_logits = self.decoder2class(hidden_states)
        compare_logits = self.decoder2compare_class(hidden_states)
        # bsz, q_num, clss_num

        logits = self.metric_3(torch.tanh(
            self.metric_1(hidden_states).unsqueeze(2) + self.metric_2(
                encoder_hidden_states
            ).unsqueeze(1)
        ))

        sub_start_logits = logits[:, :, :, 0]
        sub_end_logits = logits[:, :, :, 1]
        obj_start_logits = logits[:, :, :, 2]
        obj_end_logits = logits[:, :, :, 3]
        aspect_start_logits = logits[:, :, :, 4]
        aspect_end_logits = logits[:, :, :, 5]
        # opinion_start_logits = logits[:, :, :, 6]
        # opinion_end_logits = logits[:, :, :, 7]

        return hidden_states, compare_logits, sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits, \
            aspect_start_logits, aspect_end_logits

class SetDecoder_absa(nn.Module):
    def __init__(self, config, num_generated_triples, num_layers, num_classes, return_intermediate=False):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_generated_triples = num_generated_triples
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.query_embed = nn.Embedding(num_generated_triples, config.hidden_size)
        self.decoder2class = nn.Linear(config.hidden_size, num_classes)
        # self.decoder2span = nn.Linear(config.hidden_size, 4)

        self.metric_1 = nn.Linear(config.hidden_size, config.hidden_size * 8)
        self.metric_2 = nn.Linear(config.hidden_size, config.hidden_size * 8)
        self.metric_3 = nn.Linear(config.hidden_size * 8, 1 * 8, bias=False)
        

        torch.nn.init.orthogonal_(self.metric_1.weight, gain=1)
        torch.nn.init.orthogonal_(self.metric_2.weight, gain=1)
        torch.nn.init.orthogonal_(self.metric_3.weight, gain=1)

        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)



    def forward(self, encoder_hidden_states, encoder_attention_mask):
        """
        encoder_hidden_states: [bsz, enc_len, hidden]
        encoder_attention_mask: [bsz, enc_len, enc_len]
        """
        bsz = encoder_hidden_states.size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        # bsz, q_num, hidden
        hidden_states = self.dropout(self.LayerNorm(hidden_states))
        all_hidden_states = ()
        for i, layer_module in enumerate(self.layers):
            if self.return_intermediate:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

        # bsz, q_num, hidden
        class_logits = self.decoder2class(hidden_states)
        # bsz, q_num, clss_num

        logits = self.metric_3(torch.tanh(
            self.metric_1(hidden_states).unsqueeze(2) + self.metric_2(
                encoder_hidden_states
            ).unsqueeze(1)
        ))

        # sub_start_logits = logits[:, :, :, 0]
        # sub_end_logits = logits[:, :, :, 1]
        # obj_start_logits = logits[:, :, :, 2]
        # obj_end_logits = logits[:, :, :, 3]
        aspect_start_logits = logits[:, :, :, 4]
        aspect_end_logits = logits[:, :, :, 5]
        opinion_start_logits = logits[:, :, :, 6]
        opinion_end_logits = logits[:, :, :, 7]

        return hidden_states, class_logits, aspect_start_logits, aspect_end_logits, opinion_start_logits, opinion_end_logits

class SetDecoder_TL_After(nn.Module):
    def __init__(self, config, num_generated_triples, num_layers, num_classes, return_intermediate=False):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_generated_triples = num_generated_triples
        
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.query_embed = nn.Embedding(num_generated_triples, config.hidden_size)
        self.decoder2class = nn.Linear(config.hidden_size, num_classes)

        self.metric_1 = nn.Linear(config.hidden_size, config.hidden_size * 8)
        self.metric_2 = nn.Linear(config.hidden_size, config.hidden_size * 8)
        self.metric_3 = nn.Linear(config.hidden_size * 8, 1 * 8, bias=False)

        torch.nn.init.orthogonal_(self.metric_1.weight, gain=1) # 正交初始化方法（orthogonal_）
        torch.nn.init.orthogonal_(self.metric_2.weight, gain=1) # gain=1指定了初始化时的增益因子，这个因子影响了参数初始化的尺度大小。
        torch.nn.init.orthogonal_(self.metric_3.weight, gain=1) # gain默认为1.0
        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)

    def forward(self, encoder_hidden_states, encoder_attention_mask):
        """
        encoder_hidden_states: [bsz, enc_len, hidden]
        encoder_attention_mask: [bsz, enc_len, enc_len]
        """
        bsz = encoder_hidden_states.size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        # bsz, q_num, hidden
        hidden_states = self.dropout(self.LayerNorm(hidden_states)) 
        all_hidden_states = ()
        for i, layer_module in enumerate(self.layers):
            if self.return_intermediate:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

        # bsz, q_num, hidden
        class_logits = self.decoder2class(hidden_states)
        # bsz, q_num, clss_num

        logits = self.metric_3(torch.tanh(
            self.metric_1(hidden_states).unsqueeze(2) + self.metric_2(
                encoder_hidden_states
            ).unsqueeze(1)
        )) # torch.Size([4, 60, 512, 8])

        # unsqueeze:用于在指定位置添加维度， 例如unseueeze(1)表示在原张量的第1维后面添加一个新的维度，
        sub_start_logits = logits[:, :, :, 0]
        sub_end_logits = logits[:, :, :, 1]
        obj_start_logits = logits[:, :, :, 2]
        obj_end_logits = logits[:, :, :, 3]
        aspect_start_logits = logits[:, :, :, 4]
        aspect_end_logits = logits[:, :, :, 5]
        opinion_start_logits = logits[:, :, :, 6] ####
        opinion_end_logits = logits[:, :, :, 7] ####

        return class_logits, sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits, \
            aspect_start_logits, aspect_end_logits, opinion_start_logits, opinion_end_logits
    
class SetDecoder_TL_Before(nn.Module):
    def __init__(self, config, num_generated_triples, num_layers, num_classes, return_intermediate=False):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_generated_triples = num_generated_triples
        
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.query_embed = nn.Embedding(num_generated_triples, config.hidden_size)
        self.decoder2compareclass = nn.Linear(config.hidden_size, num_classes)
        self.metric_1 = nn.Linear(config.hidden_size, config.hidden_size * 8)
        self.metric_2 = nn.Linear(config.hidden_size, config.hidden_size * 8)
        self.metric_3 = nn.Linear(config.hidden_size * 8, 1 * 8, bias=False)

        torch.nn.init.orthogonal_(self.metric_1.weight, gain=1) # 正交初始化方法（orthogonal_）
        torch.nn.init.orthogonal_(self.metric_2.weight, gain=1) # gain=1指定了初始化时的增益因子，这个因子影响了参数初始化的尺度大小。
        torch.nn.init.orthogonal_(self.metric_3.weight, gain=1) # gain默认为1.0
        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)

    def forward(self, encoder_hidden_states, encoder_attention_mask):
        """
        encoder_hidden_states: [bsz, enc_len, hidden]
        encoder_attention_mask: [bsz, enc_len, enc_len]
        """
        bsz = encoder_hidden_states.size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        # bsz, q_num, hidden
        hidden_states = self.dropout(self.LayerNorm(hidden_states)) 
        all_hidden_states = ()
        for i, layer_module in enumerate(self.layers):
            if self.return_intermediate:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

        compare_logits = self.decoder2compareclass(hidden_states)

        logits = self.metric_3(torch.tanh(
            self.metric_1(hidden_states).unsqueeze(2) + self.metric_2(
                encoder_hidden_states
            ).unsqueeze(1)
        )) # torch.Size([4, 60, 512, 8])

        # unsqueeze:用于在指定位置添加维度， 例如unseueeze(1)表示在原张量的第1维后面添加一个新的维度，
        sub_start_logits = logits[:, :, :, 0]
        sub_end_logits = logits[:, :, :, 1]
        obj_start_logits = logits[:, :, :, 2]
        obj_end_logits = logits[:, :, :, 3]
        aspect_start_logits = logits[:, :, :, 4]
        aspect_end_logits = logits[:, :, :, 5]
        opinion_start_logits = logits[:, :, :, 6] ####
        opinion_end_logits = logits[:, :, :, 7] ####

        return compare_logits, sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits, \
            aspect_start_logits, aspect_end_logits, opinion_start_logits, opinion_end_logits

class SetDecoder_TL(nn.Module):
    def __init__(self, config, num_generated_triples, num_layers, num_classes, return_intermediate=False):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_generated_triples = num_generated_triples
        
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.query_embed = nn.Embedding(num_generated_triples, config.hidden_size)
        self.decoder2class = nn.Linear(config.hidden_size, num_classes)
        self.metric_1 = nn.Linear(config.hidden_size, config.hidden_size * 8)
        self.metric_2 = nn.Linear(config.hidden_size, config.hidden_size * 8)
        self.metric_3 = nn.Linear(config.hidden_size * 8, 1 * 8, bias=False)

        ## add
        self.layers2 = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.LayerNorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.query_embed2 = nn.Embedding(num_generated_triples, config.hidden_size)
        self.decoder2compareclass = nn.Linear(config.hidden_size, num_classes)
        self.metric_4 = nn.Linear(config.hidden_size, config.hidden_size * 8)
        self.metric_5 = nn.Linear(config.hidden_size, config.hidden_size * 8)
        self.metric_6 = nn.Linear(config.hidden_size * 8, 1 * 8, bias=False)

        torch.nn.init.orthogonal_(self.metric_1.weight, gain=1) # 正交初始化方法（orthogonal_）
        torch.nn.init.orthogonal_(self.metric_2.weight, gain=1) # gain=1指定了初始化时的增益因子，这个因子影响了参数初始化的尺度大小。
        torch.nn.init.orthogonal_(self.metric_3.weight, gain=1) # gain默认为1.0
        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)

    def forward(self, encoder_hidden_states, encoder_attention_mask):
        """
        encoder_hidden_states: [bsz, enc_len, hidden]
        encoder_attention_mask: [bsz, enc_len, enc_len]
        """
        bsz = encoder_hidden_states.size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        # bsz, q_num, hidden
        hidden_states = self.dropout(self.LayerNorm(hidden_states)) 
        all_hidden_states = ()
        for i, layer_module in enumerate(self.layers):
            if self.return_intermediate:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

        class_logits = self.decoder2class(hidden_states)

        logits = self.metric_3(torch.tanh(
            self.metric_1(hidden_states).unsqueeze(2) + self.metric_2(
                encoder_hidden_states
            ).unsqueeze(1)
        )) # torch.Size([4, 60, 512, 8])

        # add
        hidden_states2 = self.query_embed2.weight.unsqueeze(0).repeat(bsz, 1, 1)
        # bsz, q_num, hidden
        hidden_states2 = self.dropout2(self.LayerNorm(hidden_states2)) 
        all_hidden_states2 = ()
        for i, layer_module2 in enumerate(self.layers2):
            if self.return_intermediate:
                all_hidden_states2 = all_hidden_states2 + (hidden_states2,)
            layer_outputs2 = layer_module2(
                hidden_states2, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states2 = layer_outputs2[0]

        compare_logits = self.decoder2compareclass(hidden_states2)

        logits2 = self.metric_6(torch.tanh(
            self.metric_4(hidden_states).unsqueeze(2) + self.metric_5(
                encoder_hidden_states
            ).unsqueeze(1)
        )) # torch.Size([4, 60, 512, 8])

        # unsqueeze:用于在指定位置添加维度， 例如unseueeze(1)表示在原张量的第1维后面添加一个新的维度，
        sub_start_logits = logits2[:, :, :, 0]
        sub_end_logits = logits2[:, :, :, 1]
        obj_start_logits = logits2[:, :, :, 2]
        obj_end_logits = logits2[:, :, :, 3]
        aspect_start_logits = logits[:, :, :, 4]
        aspect_end_logits = logits[:, :, :, 5]
        opinion_start_logits = logits[:, :, :, 6] ####
        opinion_end_logits = logits[:, :, :, 7] ####

        return class_logits, sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits, \
            aspect_start_logits, aspect_end_logits, opinion_start_logits, opinion_end_logits

class SetDecoder(nn.Module):
    def __init__(self, config, num_generated_triples, num_layers, num_classes, return_intermediate=False):
        super().__init__()
        self.return_intermediate = return_intermediate
        self.num_generated_triples = num_generated_triples
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(num_layers)])
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.query_embed = nn.Embedding(num_generated_triples, config.hidden_size)
        self.decoder2class = nn.Linear(config.hidden_size, num_classes)
        # self.decoder2span = nn.Linear(config.hidden_size, 4)

        self.metric_1 = nn.Linear(config.hidden_size, config.hidden_size * 8)
        self.metric_2 = nn.Linear(config.hidden_size, config.hidden_size * 8)
        self.metric_3 = nn.Linear(config.hidden_size * 8, 1 * 8, bias=False)

        torch.nn.init.orthogonal_(self.metric_1.weight, gain=1) # 正交初始化方法（orthogonal_）
        torch.nn.init.orthogonal_(self.metric_2.weight, gain=1) # gain=1指定了初始化时的增益因子，这个因子影响了参数初始化的尺度大小。
        torch.nn.init.orthogonal_(self.metric_3.weight, gain=1) # gain默认为1.0
        torch.nn.init.orthogonal_(self.query_embed.weight, gain=1)

    def forward(self, encoder_hidden_states, encoder_attention_mask):
        """
        encoder_hidden_states: [bsz, enc_len, hidden]
        encoder_attention_mask: [bsz, enc_len, enc_len]
        """
        bsz = encoder_hidden_states.size()[0]
        hidden_states = self.query_embed.weight.unsqueeze(0).repeat(bsz, 1, 1)
        # bsz, q_num, hidden
        hidden_states = self.dropout(self.LayerNorm(hidden_states)) 
        all_hidden_states = ()
        for i, layer_module in enumerate(self.layers):
            if self.return_intermediate:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states, encoder_hidden_states, encoder_attention_mask
            )
            hidden_states = layer_outputs[0]

        # bsz, q_num, hidden
        class_logits = self.decoder2class(hidden_states)
        # bsz, q_num, clss_num

        logits = self.metric_3(torch.tanh(
            self.metric_1(hidden_states).unsqueeze(2) + self.metric_2(
                encoder_hidden_states
            ).unsqueeze(1)
        )) # torch.Size([4, 60, 512, 8])

        # unsqueeze:用于在指定位置添加维度， 例如unseueeze(1)表示在原张量的第1维后面添加一个新的维度，
        sub_start_logits = logits[:, :, :, 0]
        sub_end_logits = logits[:, :, :, 1]
        obj_start_logits = logits[:, :, :, 2]
        obj_end_logits = logits[:, :, :, 3]
        aspect_start_logits = logits[:, :, :, 4]
        aspect_end_logits = logits[:, :, :, 5]
        opinion_start_logits = logits[:, :, :, 6]
        opinion_end_logits = logits[:, :, :, 7]

        return class_logits, sub_start_logits, sub_end_logits, obj_start_logits, obj_end_logits, \
            aspect_start_logits, aspect_end_logits, opinion_start_logits, opinion_end_logits

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_attention_mask
    ):
        # hidden_states: bsz, q_num, hidden
        self_attention_outputs = self.attention(hidden_states)
        attention_output = self_attention_outputs[0]
        # bsz, q_num, q_num
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    encoder_hidden_shape, encoder_attention_mask.shape
                )
            )
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        cross_attention_outputs = self.crossattention(
            hidden_states=attention_output, encoder_hidden_states=encoder_hidden_states,  encoder_attention_mask=encoder_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        # bsz, q_num, hidden
        outputs = (layer_output,) + outputs
        return outputs