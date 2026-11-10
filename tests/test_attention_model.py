"""Shape, masking invariance, gradient flow, and checkpoint round-trip tests."""
import copy
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from torch import nn

from src.core.hierarchical_ssm import HierarchicalStateSpaceModel, ScalarEncoder
from src.core.module import trajectory_errors
from src.data.loading import build_datasets

ROOT = Path(__file__).resolve().parents[1]


def make_model():
    config = yaml.safe_load((ROOT/'configs/tongji_curriculum_pilot.yaml').read_text())
    normalization = {
        "feature_mean": [0.0] * 5,
        "feature_std": [1.0] * 5,
        "static_mean": [0.0] * 3,
        "static_std": [1.0] * 3,
        "condition_mean": [0.0] * 4,
        "condition_std": [1.0] * 4,
    }
    return HierarchicalStateSpaceModel([2,4,3,1,1],config['model'],normalization)


def make_batch():
    torch.manual_seed(7)
    b,n,t = 2,3,7
    point = torch.zeros(b,n,t,dtype=torch.bool)
    point[0,0,:3]=True;point[0,1,:5]=True;point[0,2,:]=True
    point[1,0,:4]=True;point[1,1,:6]=True
    valid = point.unsqueeze(-1).expand(-1,-1,-1,5).clone()
    valid[...,4] = False
    visible = point.unsqueeze(-1).expand(-1,-1,-1,5).clone()
    visible[0,0,:,1] = False
    return {
        "values": torch.randn(b, n, t, 5),
        "value_valid": valid,
        "value_visible": visible,
        "point_valid": point,
        "time_in_cycle_s": torch.arange(t).double().expand(b, n, t),
        "cycle_id": torch.tensor([[2, 7, 10], [2, 8, 0]]),
        "last_cycle_id": torch.tensor([10, 8]),
        "static_categorical": torch.tensor([[1, 1, 1, 0, 0], [1, 2, 2, 0, 0]]),
        "static_numeric": torch.tensor([[3.5, 4.2, 2.65], [3.5, 4.2, 2.5]]),
        "static_valid": torch.tensor(
            [[1, 1, 1, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 1, 1, 1]], dtype=torch.bool
        ),
        "history_u": torch.ones(b, 3, 4),
        "history_u_valid": torch.ones(b, 3, 4, dtype=torch.bool),
        "future_u": torch.ones(b, 4, 4),
        "future_u_valid": torch.ones(b, 4, 4, dtype=torch.bool),
    }


class AttentionModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_shapes_and_each_module_receives_finite_gradients(self):
        model=make_model()
        batch=make_batch()
        prediction,states=model(batch,return_states=True)
        self.assertEqual(tuple(prediction.shape),(2,4))
        self.assertEqual(tuple(states.shape),(2,4,16))
        encoder = model.observation_encoders[0]
        assert isinstance(encoder, ScalarEncoder)
        first, second = encoder.network[0], encoder.network[2]
        assert isinstance(first, nn.Linear) and isinstance(second, nn.Linear)
        self.assertEqual(tuple(first.weight.shape), (8, 1))
        self.assertEqual(tuple(second.weight.shape), (4, 8))
        target=torch.full_like(prediction,.8)
        valid=torch.ones_like(target,dtype=torch.bool)
        valid[:,1]=False;target[:,1]=float('nan')
        mse,_=trajectory_errors(prediction,target,valid)
        mse.mean().backward()
        for prefix in ['observation_encoders','categorical_encoders','numeric_encoders','point_blocks',
                       'cycle_blocks','dynamics','head']:
            gradients=[p.grad for name,p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
            self.assertTrue(gradients,prefix)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients),prefix)
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0,prefix)
        unk_grad = model.observation_encoders[4].get_parameter("unk").grad
        assert unk_grad is not None
        self.assertGreater(float(unk_grad.abs().sum()), 0)
        pointers = [encoder.get_parameter("unk").data_ptr() for encoder in model.observation_encoders]
        self.assertEqual(len(set(pointers)), 5)
        pointers = [encoder.get_parameter("weight").data_ptr() for encoder in model.categorical_encoders]
        self.assertEqual(len(set(pointers)), 5)

    def test_padding_missing_and_hidden_numbers_do_not_change_output(self):
        model=make_model().eval()
        batch=make_batch()
        with torch.no_grad():
            expected=model(batch)
            changed=copy.deepcopy(batch)
            invisible=~(changed['value_valid'] & changed['value_visible'] & changed['point_valid'].unsqueeze(-1))
            changed['values'][invisible]=float('nan')
            changed['time_in_cycle_s']=changed['time_in_cycle_s'].clone()
            changed['time_in_cycle_s'][~changed['point_valid']]=float('nan')
            changed['cycle_id'][~changed['point_valid'].any(-1)]=999999
            torch.testing.assert_close(expected,model(changed),atol=1e-6,rtol=1e-5)
            # Extra point PAD and an extra empty cycle must not change the state.
            for key in ['values','value_valid','value_visible']:
                x=changed[key]
                extra=x.new_zeros(x.shape[0],x.shape[1],2,x.shape[3])
                x=torch.cat([x,extra],dim=2)
                changed[key]=torch.cat([x,x.new_zeros(x.shape[0],1,x.shape[2],x.shape[3])],dim=1)
            for key in ['point_valid','time_in_cycle_s']:
                x=changed[key]
                x=torch.cat([x,x.new_zeros(x.shape[0],x.shape[1],2)],dim=2)
                changed[key]=torch.cat([x,x.new_zeros(x.shape[0],1,x.shape[2])],dim=1)
            changed['cycle_id']=torch.cat([changed['cycle_id'],torch.zeros(2,1,dtype=torch.int64)],dim=1)
            torch.testing.assert_close(expected,model(changed),atol=1e-6,rtol=1e-5)

    def test_early_fusion_has_equal_parameters_and_masks_padding(self):
        late=make_model().eval()
        early=make_model().eval()
        early.config=dict(early.config,static_fusion='early')
        early.load_state_dict(late.state_dict())
        self.assertEqual(sum(p.numel() for p in late.parameters()),sum(p.numel() for p in early.parameters()))
        batch=make_batch()
        with torch.no_grad():
            baseline=early(batch)
            self.assertEqual(tuple(baseline.shape),(2,4))
            self.assertFalse(torch.allclose(baseline,late(batch)))
            noisy=copy.deepcopy(batch)
            hidden=~(noisy['value_valid'] & noisy['value_visible'] & noisy['point_valid'].unsqueeze(-1))
            noisy['values'][hidden]=float('nan')
            torch.testing.assert_close(baseline,early(noisy),atol=1e-6,rtol=1e-5)
        early.train()
        early(batch).square().mean().backward()
        weight_grad = early.categorical_encoders[0].get_parameter("weight").grad
        assert weight_grad is not None
        self.assertGreater(float(weight_grad.abs().sum()), 0)

    def test_state_dict_round_trip(self):
        model=make_model().eval();batch=make_batch()
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt'
            torch.save(model.state_dict(),path)
            restored=make_model().eval()
            restored.load_state_dict(torch.load(path,weights_only=True))
            with torch.no_grad():
                torch.testing.assert_close(model(batch),restored(batch))

    @unittest.skipUnless((ROOT/'data/processed/tongji_v1/manifest.json').exists(),'Preprocessed data required')
    def test_real_cell_schedule_and_holdout_split(self):
        config=yaml.safe_load((ROOT/'configs/tongji_curriculum_pilot.yaml').read_text())
        config['data']['root']=str(ROOT/'data/processed/tongji_v1')
        datasets,splits=build_datasets(config)
        try:
            cell_id = splits["train_cells"][0]
            train_split = splits["train_cycle_positions"]
            val_split = splits["val_cycle_positions"]
            assert train_split is not None and val_split is not None
            train_cycle_positions = train_split[cell_id]
            val_cycle_positions = val_split[cell_id]
            self.assertFalse(set(train_cycle_positions) & set(val_cycle_positions))
            self.assertEqual(sorted(train_cycle_positions + val_cycle_positions), [1, 2, 3, 4, 5, 6])
            train = datasets["train"]
            self.assertEqual(
                [train[i, 1]["last_cycle_position"] for i in range(len(train))], train_cycle_positions
            )
            self.assertFalse(set(splits["train_cells"]) & set(splits["test_cells"]))
        finally:
            for ds in datasets.values():ds.close()


if __name__=='__main__':
    unittest.main()
