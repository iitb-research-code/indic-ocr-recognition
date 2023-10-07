import torch
import torch.nn as nn

from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Stack
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class T5DecoderOnlyForCausalLM(T5PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder = T5Stack(config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.is_decoder = True
        config.is_decoder = True
        config.use_cache = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        decoder_input_ids=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        head_mask=None,
        use_cache=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = decoder_outputs.last_hidden_state
        logits = self.lm_head(last_hidden_state)
        hidden_states = decoder_outputs.hidden_states
        past_key_values = decoder_outputs.past_key_values
        attentions = decoder_outputs.attentions
        cross_attentions = decoder_outputs.cross_attentions

        return CausalLMOutputWithCrossAttentions(
            logits = logits,
            past_key_values = past_key_values,
            hidden_states = hidden_states,
            attentions = attentions,
            cross_attentions = cross_attentions
        )


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)
