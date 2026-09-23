"""Split at block 12 and keep differentiable projected KV until the next full step."""
import torch
from transformers import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    create_causal_mask,torch_causal_conv1d_update,torch_recurrent_gated_delta_rule,torch_chunk_gated_delta_rule)
from ced_training.engine import load_runner
from ced_training.protocol import sha256
from .protocol import SEMANTICS

class PairCache(DynamicCache):
    def update_recurrent_state(self,state,layer_idx,**kwargs):
        layer=self.layers[layer_idx]
        if not layer.is_recurrent_states_initialized:layer.lazy_initialization(recurrent_states=state)
        # Replacing rather than copy_ preserves the prior state saved for backward.
        layer.recurrent_states=state
        return state

def detach_cache(cache):
    for layer in cache.layers:
        for name in ('keys','values','conv_states','recurrent_states'):
            value=getattr(layer,name,None)
            if value is not None:setattr(layer,name,value.detach())

class Runtime:
    def __init__(self,root,config,data,checkpoint=None):
        self.config=config;self.runner=load_runner(root,config)
        self.model=self.runner.model;self.lm=self.runner.lm
        weights=list((root/config['model_path']).glob('*.safetensors'))
        if len(weights)!=1 or sha256(weights[0])!=data.manifest['model_weights_sha256']:raise ValueError('Teacher weights differ')
        # The local torch path supports input gradients; do not rely on inference-only fused kernels.
        for layer in self.lm.layers:
            if hasattr(layer,'linear_attn'):
                m=layer.linear_attn;m.causal_conv1d_fn=None;m.causal_conv1d_update=torch_causal_conv1d_update
                m.chunk_gated_delta_rule=torch_chunk_gated_delta_rule;m.recurrent_gated_delta_rule=torch_recurrent_gated_delta_rule
        self.threshold=None;self.loaded=None;self.source_interval=None
        source=checkpoint or root/config['warm_start']
        state=torch.load(source,map_location='cpu',weights_only=True)
        if state.get('semantics')==SEMANTICS:
            if state['manifest_sha256']!=data.digest or state['config']!=config:raise ValueError('Checkpoint/config/data mismatch')
            self.loaded=state
            for name in ('readout_map','kv_projectors','confidence_head'):getattr(self.runner,name).load_state_dict(state['modules'][name])
            self.threshold=state.get('threshold')
        else:
            if checkpoint is not None:raise ValueError('Expected Top8 checkpoint')
            if state.get('semantics') not in ('top8_tail_teacher_forced_qualified_exit_immediate_full_v1','block12_residual_to_target_self_attn_raw_kv_v1'):raise ValueError('Unsupported warm-start semantics')
            if state.get('manifest_sha256',data.digest)!=data.digest:raise ValueError('Warm-start dataset differs')
            if state.get('stage') not in ('modules','confidence','calibrated') or state.get('model_weights_sha256')!=data.manifest['model_weights_sha256']:raise ValueError('Invalid initial weights')
            for name in ('readout_map','kv_projectors','confidence_head'):getattr(self.runner,name).load_state_dict(state['modules'][name])
        self.source_interval=state.get('interval')
        self.source=str(source);self.source_sha256=sha256(source)
        self.initialization={name:all(torch.equal(value.detach().cpu(),state['modules'][name][key])
            for key,value in getattr(self.runner,name).state_dict().items())
            for name in ('readout_map','kv_projectors','confidence_head')}
        if not all(self.initialization.values()):raise AssertionError('Warm-start parameters differ from source')
        self.runner.requires_grad_(False)
    def cache(self):return PairCache(config=self.model.config)
    @torch.no_grad()
    def prefill(self,tokens):
        cache=self.cache()
        logits=self.runner.native(tokens,cache)[:,-1,:]
        return cache,logits
    def begin(self,token,cache):
        hidden=self.lm.embed_tokens(token)
        positions=torch.arange(token.shape[1],device=token.device)+cache.get_seq_length()
        positions=positions.view(1,1,-1).expand(4,token.shape[0],-1)
        pos=positions[0]
        causal=create_causal_mask(config=self.model.config.text_config,inputs_embeds=hidden,attention_mask=None,past_key_values=cache,position_ids=pos)
        linear=self.lm._update_linear_attn_mask(None,cache)
        rope=self.lm.rotary_emb(hidden,positions[1:])
        state={'cache':cache,'pos':pos,'causal':causal,'linear':linear,'rope':rope}
        for i in range(12):hidden=self.layer(i,hidden,state)
        return hidden,state
    def layer(self,i,hidden,state):
        return self.lm.layers[i](hidden,position_embeddings=state['rope'],attention_mask=state['causal'] if (i+1)%4==0 else state['linear'],
                                  position_ids=state['pos'],past_key_values=state['cache'],use_cache=True)
    def finish(self,hidden,state,compute_logits=True):
        for i in range(12,24):hidden=self.layer(i,hidden,state)
        if not compute_logits:return None
        return self.model.lm_head(self.lm.norm(hidden[:,-1:,:]))[:,-1,:]
    def readout(self,hidden):return self.model.lm_head(self.runner.readout_map(hidden))[:,-1,:]
    def readout_all(self,hidden):
        return self.model.lm_head(self.runner.readout_map(hidden)).reshape(-1,self.model.config.text_config.vocab_size)
    def deep_context(self,state,cache,index):
        # One unpadded query can see every accumulated key. The shallow cache has
        # advanced by a chunk, but the deep cache contains only preceding positions.
        return {'cache':cache,'pos':state['pos'][:,index:index+1],'causal':None,'linear':None,
                'rope':tuple(x[:,index:index+1] for x in state['rope'])}
    def project(self,hidden,state):self.runner.project_cache(hidden,state['cache'],state['rope'],None)
    def gate(self,hidden):return self.runner.confidence_head(hidden.float()).reshape(-1)
    def adaptive(self,token,cache):
        h,state=self.begin(token,cache)
        score=self.gate(h).sigmoid().item()
        accepted=self.threshold is not None and score>=self.threshold
        if accepted:
            logits=self.readout(h);self.project(h,state)
        else:logits=self.finish(h,state)
        return logits,accepted
    def state(self):
        return {name:{k:v.detach().cpu() for k,v in getattr(self.runner,name).state_dict().items()}
                for name in ('readout_map','kv_projectors','confidence_head')}
