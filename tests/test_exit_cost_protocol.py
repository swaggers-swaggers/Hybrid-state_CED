"""Stdlib checks; never import a model runtime or CUDA."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from exit_cost.protocol import choose_exit,make_plan,percentile,summarize

class ProtocolTests(unittest.TestCase):
    def setUp(self):self.config=json.loads((ROOT/'configs/qwen35_08b_exit_cost.json').read_text())
    def test_default_workload(self):
        p=make_plan(self.config)
        self.assertEqual(p['timed_decode_steps'],5760)
        self.assertEqual(p['warmup_decode_steps'],144)
        self.assertEqual(p['profile_decode_steps'],144)
        self.assertEqual(p['estimated_minutes'],[2.5,7.0])
        self.assertEqual(p['projection']['targets'],[16,20,24])
        self.assertEqual(p['projection']['independent_modules'],3)
        self.assertEqual([c['name'] for c in p['scenarios']],['full','gate_reject','exit_12'])
    def test_adaptive_explicit(self):
        self.config['include_adaptive']=True;p=make_plan(self.config)
        self.assertEqual(p['timed_decode_steps'],7680)
        self.assertEqual(p['scenarios'][-1]['policy'],'network')
        self.assertFalse(p['confidence']['trained'])
    def test_threshold_and_control_overrides(self):
        self.assertFalse(choose_exit(.89,'network',.9))
        self.assertTrue(choose_exit(.9,'network',.9))
        self.assertTrue(choose_exit(.1,'force_accept',.9))
        self.assertFalse(choose_exit(.99,'force_reject',.9))
        for score in [float('nan'),float('inf'),-1,2]:
            with self.assertRaises(ValueError):choose_exit(score,'network',.9)
    def test_incompatible_config_rejected(self):
        for k,v in [('exit_depth',8),('kv_targets',[12,16,20,24]),('top_k',5),('head_depths',[12]),('decode_tokens',0),('context_lengths',[]),('confidence_threshold',float('nan')),('include_adaptive','yes')]:
            with self.subTest(key=k):
                c=copy.deepcopy(self.config);c[k]=v
                with self.assertRaises(ValueError):make_plan(c)
    def test_pairing_and_exit_rate(self):
        rows=[]
        for name,repeat,latency in [('exit_12',1,20),('full',0,20),('exit_12',0,10),('full',1,60)]:
            rows.append(dict(context=256,variant=0,repeat=repeat,scenario=name,wall_ms_per_token=latency,gpu_ms_per_token=latency-1,peak_allocated_mib=1000,exit_count=64 if name=='exit_12' else 0,decode_tokens=64))
        report=summarize(rows)
        self.assertEqual(report[0]['median_paired_speedup'],1)
        self.assertEqual(report[1]['median_paired_speedup'],2.5)
        self.assertEqual(report[0]['exit_rate'],0)
        self.assertEqual(report[1]['exit_rate'],1)
        self.assertAlmostEqual(percentile([10,20],.95),19.5)
    def test_default_cli_never_imports_runtime(self):
        code="""
import runpy,sys
class Guard:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','transformers') or fullname in ('exit_cost.runtime','exit_cost.cache'):
            raise AssertionError('Model import in plan mode: '+fullname)
sys.meta_path.insert(0,Guard())
sys.argv=[sys.argv[1]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
        p=subprocess.run([sys.executable,'-c',code,str(ROOT/'scripts/benchmark_exit_cost.py')],check=True,capture_output=True,text=True)
        self.assertEqual(json.loads(p.stdout)['status'],'PLAN_ONLY_NO_MODEL_LOADED')
if __name__=='__main__':unittest.main()
