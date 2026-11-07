"""Study-level split, metric weighting, and environment hygiene checks."""
from pathlib import Path
import os
import unittest
from unittest.mock import patch
import torch
from training.study import clean_environment,evaluate_cells,stratified_cell_split

ROOT=Path(__file__).resolve().parents[1]


class StudyTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'data/tongji_v1/manifest.json').exists(),'Processed data required')
    def test_all_cells_are_disjoint_and_protocol_stratified(self):
        split=stratified_cell_split(ROOT/'data/tongji_v1')
        self.assertEqual(split,stratified_cell_split(ROOT/'data/tongji_v1'))
        all_cells=[cid for name in ['train','val','test'] for cid in split[name]]
        self.assertEqual(len(all_cells),130)
        self.assertEqual(len(set(all_cells)),130)
        for parts in split['groups'].values():
            self.assertTrue(all(parts[name] for name in ['train','val','test']))

    def test_task_environment_excludes_credentials(self):
        with patch.dict(os.environ,{'EXAMPLE_API_KEY':'dummy-value','HF_TOKEN':'dummy-value'}):
            env=clean_environment()
        self.assertNotIn('EXAMPLE_API_KEY',env)
        self.assertNotIn('HF_TOKEN',env)
        self.assertEqual(env['WANDB_MODE'],'offline')

    def test_macro_metric_does_not_weight_long_cells_more(self):
        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__();self.parameter=torch.nn.Parameter(torch.zeros(1))
            def forward(self,batch):return batch['prediction']
        batch=dict(cell_id=['short','long','long','long'],prediction=torch.zeros(4,1),
                   target_soh=torch.tensor([[1.],[0.],[0.],[0.]]),target_valid=torch.ones(4,1,dtype=torch.bool),
                   anchor_cycle=torch.arange(4),target_cycle_id=torch.arange(4).view(4,1))
        metrics,_=evaluate_cells(Dummy(),[batch])
        self.assertAlmostEqual(metrics['mse'],.25)
        self.assertAlmostEqual(metrics['cell_macro_mse'],.5)


if __name__=='__main__':unittest.main()
