"""Execute actual dispatch methods with doubles; no tensor numerics or model imports."""
import ast
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from exit_cost.protocol import ATTENTION_DEPTHS,EXIT_DEPTH,choose_exit

def load_method(class_name,method,namespace):
    tree=ast.parse((ROOT/'exit_cost/runtime.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==method)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'runtime_method','exec'),namespace)
    return namespace[method]

class TensorDouble:
    shape=(1,1,1024)
    device='fake'
    def __init__(self,name='tensor',score=0):self.name=name;self.score=score
    def __getitem__(self,index):return self
    def __add__(self,x):return self
    def view(self,*args):return self
    def expand(self,*args):return self
    def float(self):return self
    def sigmoid(self):return self
    def item(self):return self.score
    def argmax(self,**kwargs):return TensorDouble('output_token')

class DispatchTests(unittest.TestCase):
    def run_step(self,policy,score):
        events=[]
        def block(depth):
            def call(hidden,**kwargs):events.append(f'block_{depth}');return TensorDouble(f'h{depth}')
            return call
        def confidence(hidden):
            self.assertEqual(hidden.name,'h12');events.append('confidence');return TensorDouble(score=score)
        def mapping(hidden):
            self.assertEqual(hidden.name,'h12');events.append('map');return TensorDouble('mapped')
        def vocab(hidden):events.append('vocabulary');return TensorDouble('logits')
        def norm(hidden):
            self.assertEqual(hidden.name,'h24');events.append('final_norm');return hidden
        def project(hidden,*args):
            self.assertEqual(hidden.name,'h12');events.extend(['P16','P20','P24'])
        lm=SimpleNamespace(embed_tokens=lambda t:TensorDouble('embed'),layers=[block(d) for d in range(1,25)],_update_linear_attn_mask=lambda *a:None,rotary_emb=lambda *a:('cos','sin'),norm=norm)
        runner=SimpleNamespace(lm=lm,model=SimpleNamespace(config=SimpleNamespace(text_config=None),lm_head=vocab),confidence_head=confidence,readout_map=mapping,project_cache=project,config={'confidence_threshold':.9})
        namespace=dict(torch=SimpleNamespace(arange=lambda *a,**k:TensorDouble()),create_causal_mask=lambda **k:None,span=lambda *a:nullcontext(),ATTENTION_DEPTHS=ATTENTION_DEPTHS,EXIT_DEPTH=EXIT_DEPTH,choose_exit=choose_exit)
        step=load_method('CostRunner','step',namespace)
        _,_,exited=step(runner,TensorDouble(),SimpleNamespace(get_seq_length=lambda:256),{'policy':policy})
        return events,exited
    def test_low_confidence_skips_intermediate_vocab_and_projects(self):
        events,exited=self.run_step('network',.2)
        self.assertFalse(exited)
        self.assertEqual(events,[f'block_{d}' for d in range(1,13)]+['confidence']+[f'block_{d}' for d in range(13,25)]+['final_norm','vocabulary'])
    def test_high_confidence_gate_precedes_vocab(self):
        events,exited=self.run_step('network',.95)
        self.assertTrue(exited)
        self.assertEqual(events,[f'block_{d}' for d in range(1,13)]+['confidence','map','vocabulary','P16','P20','P24'])
    def test_baseline_and_forced_controls(self):
        events,exited=self.run_step('disabled',1)
        self.assertFalse(exited);self.assertNotIn('confidence',events)
        self.assertEqual(len([e for e in events if e.startswith('block_')]),24)
        self.assertTrue(self.run_step('force_accept',.1)[1])
        self.assertFalse(self.run_step('force_reject',1)[1])
    def test_projector_uses_same_single_input_for_both_outputs(self):
        seen=[]
        obj=SimpleNamespace(key=lambda h:seen.append(('K',h)) or 'K_raw',value=lambda h:seen.append(('V',h)) or 'V')
        forward=load_method('KVProjector','forward',{})
        hidden=object()
        self.assertEqual(forward(obj,hidden),('K_raw','V'))
        self.assertEqual(seen,[('K',hidden),('V',hidden)])
        self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main()
