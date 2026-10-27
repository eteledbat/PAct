"""Read-only official sampling: cache frozen pooled inputs to the released MLP.

No asset is replaced. Capture four individual inference timesteps and the
official last-20-steps average. This is not a cache of GT-noised training latents.
"""
import argparse
import json
import os
import pathlib
import runpy
import shutil
import sys
import time

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from modules.pact.pipelines import PActPipeline


class CacheComplete(Exception):
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--case', required=True)
    a = p.parse_args()
    root = pathlib.Path(a.root)
    cfg = json.loads((root/'config.json').read_text())
    run = pathlib.Path(cfg.get('source_runs',cfg['runs'])[a.case])
    out = root/'features'/a.case
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    original = PActPipeline.sample_slat_arti

    def sample(self, *args, **kwargs):
        model = self.models['slat_arti_flow_model']
        captures = {}
        label = ['average']

        def capture(module, inputs):
            captures[label[0]] = inputs[0].feats.detach().float().cpu().clone()

        hook = model.art_out_layers[0].register_forward_pre_hook(capture)
        old_sample = self.slat_sampler.sample
        sampled = {}

        def sampling(*sa, **sk):
            result = old_sample(*sa, **sk)
            sampled.update(result=result,kwargs=sk)
            return result

        self.slat_sampler.sample = sampling
        try:
            slat, arti = original(self, *args, **kwargs)
            raw = arti[0].feats.detach().float().cpu().clone()
            # Extra read-only head calls occur AFTER official average regression.
            # They cannot alter the saved raw prediction or sampling trajectory.
            for index in [5,11,17,24]:
                label[0] = f'step_{index+1:02d}'
                model.fwd_articulation_info(sampled['result'].pred_arti_t[index], **sampled['kwargs'])
        finally:
            hook.remove()
            self.slat_sampler.sample = old_sample
        old_file = run/'joint_trace/raw_head_00.npy'
        old = np.load(old_file) if old_file.exists() else None
        error = float(np.max(np.abs(raw.numpy()-old))) if old is not None else None
        assert np.isfinite(raw.numpy()).all()
        if old is not None: assert raw.shape == old.shape
        state = {k: v.detach().float().cpu().clone()
                 for k, v in model.art_out_layers.state_dict().items()}
        torch.save({'pooled': captures, 'mlp_state': state, 'raw': raw}, out/'cache.pt')
        status = {'state': 'complete', 'case': a.case,
                  'original_run': str(run), 'max_raw_difference_vs_original': error,
                  'cached': list(captures), 'shape': list(captures['average'].shape),
                  'elapsed_seconds': time.monotonic()-started,
                  'geometry_replaced': False, 'network_trained': False,
                  'feature_source': 'Frozen official sampler; four single inference steps plus last-20 feature average, not GT-noised training latents'}
        (out/'feature_capture.json').write_text(json.dumps(status, indent=2))
        print(json.dumps(status), flush=True)
        return slat,arti

    PActPipeline.sample_slat_arti = sample
    os.environ['PACT_LOAD_AUDIT'] = str(out/'checkpoint_loads.jsonl')
    os.environ['PACT_JOINT_TRACE'] = str(out/'joint_trace')
    sys.argv = ['guarded_infer.py', '--model', str(REPO/'repro/cache/PAct-official'),
                '--data_dir', cfg['input_dirs'][a.case],
                '--outdir', str(out/'baseline'), '--batch_size', '1',
                '--ss_steps', '25', '--slat_steps', '25', '--ss_cfg_strength', '7',
                '--slat_cfg_strength', '7', '--arti_mean_num', '20',
                '--render_num_frames','21','--export_arti_objects','--save_gs',
                '--no-save_video_grid', '--no-save_cond_vis_grid']
    runpy.run_path(str(REPO/'repro/tools/guarded_infer.py'), run_name='__main__')
    objects=list((out/'baseline').rglob('object.json'));assert len(objects)==1
    provenance=json.loads((run/'input_provenance.json').read_text())
    shutil.copy2(run/'input_provenance.json',out/'input_provenance.json')
    cache=torch.load(out/'cache.pt',weights_only=False)
    raw=np.load(out/'joint_trace/raw_head_00.npy')
    assert np.array_equal(raw,cache['raw'].numpy()),'Same-run raw feature/geometry association failed'
    assert len(json.loads(objects[0].read_text())['diffuse_tree'])==provenance['part_count']
    status=json.loads((out/'feature_capture.json').read_text())
    status.update(object=str(objects[0]),elapsed_seconds=time.monotonic()-started,
                  same_run_feature_asset_association=True,scene=a.case,
                  geometry_replaced=False,new_original_baseline=True)
    (out/'status.json').write_text(json.dumps(status,indent=2))
    print('SAME-RUN FROZEN BASELINE COMPLETE',a.case,flush=True)


if __name__ == '__main__':
    main()
