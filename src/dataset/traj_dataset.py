import numpy as np
import torch
from torch.utils.data import Dataset
import os
import h5py
from torch_cluster import fps
import json
import random

class TrajDataset(Dataset):
    def __init__(self, split, cfg):
        
        self.cfg = cfg
        self.dataset_path = cfg.dataset_path
        self.split = split
        self.stage = cfg.stage # 'shape' or 'deform'
        self.mode = cfg.mode # 'ae' or 'diff'
        self.repeat = cfg.repeat
        self.seed = cfg.seed 
        self.pc_size = cfg.pc_size
        self.n_sample_pro_model = cfg.n_sample_pro_model
        self.n_frames_interval = cfg.n_frames_interval
        self.n_training_frames = cfg.n_training_frames
        self.input_frames = cfg.get('input_frames', 3)
        self.output_frames = cfg.get('output_frames', 3)
        # 1b: number of output chunks to reserve per window for multi-step rollout training.
        # 1 = single chunk (default, == run23 / eval). Only the training split uses >1.
        self.rollout_unroll_steps = cfg.get('rollout_unroll_steps', 1)
        # curriculum: draw window start frame at random over [0, max_start] at getitem time.
        self.rollout_random_window = cfg.get('rollout_random_window', False)
        # with random windows, force one window per model to start at 0 (restore start=0 density).
        self.rollout_force_start0 = cfg.get('rollout_force_start0', False)
        # curriculum: number of random windows to emit per model (None -> stride-5 count).
        self.windows_per_model = cfg.get('windows_per_model', None)
        # single-step aug: extra random-start windows appended on top of the fixed stride-5 grid.
        self.train_extra_random_windows = cfg.get('train_extra_random_windows', 0)
        self.batch_size = cfg.batch_size
        self.has_gravity = cfg.get('has_gravity', False)
        self.max_num_forces = cfg.get('max_num_forces', 1)
        # input=1 消融:True → start_vel 用后向差分(只用过去帧),避免中心差分偷看目标帧。
        # 默认 False = 现有臂中心差分,字节不变。
        self.causal_start_vel = cfg.get('causal_start_vel', False)

        # if os.path.exists(os.path.join(self.dataset_path, cfg.dataset_list)):
        if os.path.exists(cfg.dataset_list):
            print(f'Loading {cfg.dataset_list}')
            with open(cfg.dataset_list, 'r') as f:
                self.split_lst = json.load(f)
        else:
            self.split_lst = [f for f in sorted(os.listdir(self.dataset_path)) if f.endswith('h5')]
        random.seed(0)
        random.shuffle(self.split_lst)
        print('Number of data:', len(self.split_lst))
        
        if cfg.overfit:
            self.split_lst = self.split_lst[:1]
        elif cfg.dataset_path.endswith('_test') or cfg.dataset_list.endswith('test.json') or cfg.dataset_list.endswith('test_list.json'):
            self.split_lst = self.split_lst[:100]
            print('Test split:', self.split_lst)
        else:
            if split == 'train':
                self.split_lst = self.split_lst[:-4]
            elif split == 'test':
                # 单独 eval:只评最后 4 个 —— train 用 [:-4],这 4 个训练完全没见过,是干净 held-out
                self.split_lst = self.split_lst[-4:]
                print('Eval (clean held-out) split:', self.split_lst)
            else:  # 'val':训练过程中的验证,保持原状 [-8:],不动
                self.split_lst = self.split_lst[-8:]
                print('Val split:', self.split_lst)
        self.split_lst_save = self.split_lst.copy()
        self.split_lst_pcl_len = [25] * len(self.split_lst_save)
        # if not os.path.exists(os.path.join(self.dataset_path, f'info_deform_ae_{split}.json')):
        self.prepare_data_lst()
        # with open(os.path.join(self.dataset_path, f'info_deform_ae_{split}.json'), "w") as f:
        #     json.dump(self.models, f)
        #     print(f'Saved info_deform_ae_{split}.json')
        # else:
        #     self.models = json.load(open(os.path.join(self.dataset_path, f'info_deform_ae_{split}.json'), 'r'))
        #     print(f'Loaded info_deform_ae_{split}.json')
        
        print("Current stage: [bold red]{}[/bold red]".format(self.stage))
        print("Current mode: [bold red]{}[/bold red]".format(self.mode))
        print("Current split: [bold red]{}[/bold red]".format(self.split))
        print("Dataset is repeated [bold cyan]{}[/bold cyan] times".format(self.repeat))
        print("Length of split: {}".format(len(self.split_lst) if self.stage == 'shape' else len(self.models)))

    def prepare_data_lst(self): 
        self.models = []
        # Dynamically read actual frame counts from h5 files
        self.split_lst_pcl_len = []
        for model_name in self.split_lst_save:
            try:
                model_metas = h5py.File(os.path.join(self.dataset_path, f'{model_name}'), 'r')
                num_frames = model_metas['x'].shape[0]
                self.split_lst_pcl_len.append(num_frames)
                model_metas.close()
            except Exception as e:
                print(f"Warning: Failed to read frame count from {model_name}: {e}")
                self.split_lst_pcl_len.append(49)  # fallback to default
        
        if self.stage == 'deform':
            if self.mode == 'ae':
                if self.split == 'train':
                    models_out, indices_out = self.random_sample_indexes(self.split_lst_save * self.repeat, self.split_lst_pcl_len * self.repeat)
                    self.models += [{"model": m, "indices": indices_out[i]} for i, m in enumerate(models_out)]
                else: # Evaluate
                    for m in self.split_lst_save:
                        for i in range(1, self.batch_size + 1):
                            self.models += [{"model": m, "indices": [i-1, i]}]
            elif self.mode == 'diff':
                # 1b: reserve rollout_unroll_steps output chunks so multi-step GT exists in the window.
                required_span = (self.input_frames + self.output_frames * self.rollout_unroll_steps - 1) * self.n_frames_interval + 1
                for model_name, total_frames in zip(self.split_lst_save, self.split_lst_pcl_len):
                    max_start = total_frames - required_span
                    if max_start < 0:
                        continue
                    if self.split == 'val':
                        # 训练中每 500 步的 validation:固定取每个 model 第 1-5 帧(start_idx=0)
                        # 作为输入再 rollout,不随机采样 —— 这样各 checkpoint 的 val 视图完全可比/可复现。
                        # 只影响 val;train 仍走随机窗口,test(eval.py)仍走 stride-5 的多窗口。
                        self.models.append({"model": model_name, "start_idx": 0})
                    elif self.rollout_random_window:
                        # random-start windows (train only): draw the actual start frame over
                        # [0, max_start] at getitem time (start_idx=-1 marker). windows_per_model
                        # controls how many windows this model contributes per epoch (curriculum:
                        # 8 / 4 / 2 for K=1/2/3); falls back to the stride-5 count when unset.
                        n_win = self.windows_per_model if self.windows_per_model else len(range(0, max_start + 1, 5))
                        if self.rollout_force_start0 and n_win > 0:
                            # one fixed start=0 window per model per epoch; rest stay random.
                            self.models.append({"model": model_name, "start_idx": 0})
                            n_win -= 1
                        for _ in range(n_win):
                            self.models.append({"model": model_name, "start_idx": -1, "max_start": max_start})
                    else:
                        for start_idx in range(0, max_start + 1, 5):
                            self.models.append({"model": model_name, "start_idx": start_idx})
                        # single-step data aug (train only): append extra random-start windows on top
                        # of the fixed grid. Eval/val keep deterministic windows (this branch is also
                        # hit by eval.py's 'test' split, so gate on train).
                        if self.split == 'train' and self.train_extra_random_windows > 0:
                            for _ in range(self.train_extra_random_windows):
                                self.models.append({"model": model_name, "start_idx": -1, "max_start": max_start})
            else:
                raise NotImplementedError("mode not implemented")
    
    def __getitem__(self, index):
        if self.stage == 'deform':
            if self.mode == 'ae':
                return self.get_deform_ae(index)
            elif self.mode == 'diff':
                return self.get_deform_diff(index)

    def __len__(self):
        if self.stage == 'deform':
            if self.mode == 'ae':
                if self.split == 'train':
                    return sum(self.split_lst_pcl_len) * self.repeat
                else:
                    return len(self.split_lst_save) * self.batch_size # number of sequences
            elif self.mode == 'diff':
                return len(self.models)
            else:
                raise NotImplementedError("mode not implemented")
    
    def random_sample_indexes(self, models, models_len):
        n_sample_pro_model = self.n_sample_pro_model
        interval_between_frames = self.interval_between_frames
        n_selected_frames = self.n_selected_frames

        # Initialize output lists
        models_out = []
        indexes_out = []

        # Loop over each model
        for idx, model in enumerate(models):
            # For each sample per model
            for n in range(n_sample_pro_model):
                # Initialize indices list for current sample
                indexes = []

                # Select n_selected_frames number of indices
                for i in range(n_selected_frames):
                    # If first index, randomly select from range
                    if i == 0:
                        # indexes.append(np.random.randint(0, models_len[idx] - interval_between_frames))
                        indexes.append(np.random.randint(0, models_len[idx]))
                    else:
                        # For subsequent indices, select within interval_between_frames from the previous index
                        indexes.append( min(indexes[-1] + np.random.randint(0, interval_between_frames), models_len[idx]-1) )
                    
                # Append the selected indices and corresponding model to output lists
                indexes_out.append(sorted(indexes))
                models_out.append(model)
        
        return models_out, indexes_out  
    
    def get_deform_ae(self, index):
        model = self.models[index]
        model_name = model["model"]
        model_indices = model["indices"]

        model_info = {}
        model_info["model"] = model_name
        model_info["indices"] = model_indices

        model_metas = h5py.File(os.path.join(self.dataset_path, f'{model_name}'), 'r')
        model_pcls = torch.from_numpy(np.array(model_metas['x']))

        ind = np.random.default_rng(seed=self.seed).choice(model_pcls[0].shape[0], self.pc_size, replace=False)
        points_src = model_pcls[model_indices[0]][ind]
        points_tgt = model_pcls[model_indices[1]][ind]

        model_data = {}
        model_data['points_src'] = points_src.float()
        model_data['points_tgt'] = points_tgt.float()
        return model_data, model_info
    
    def get_deform_diff(self, index):
        
        model = self.models[index]
        model_name = model["model"]
        start_idx = model["start_idx"]
        if start_idx < 0:   # random-window mode: draw a start over [0, max_start]
            start_idx = int(np.random.randint(0, model["max_start"] + 1))

        input_indices = np.arange(start_idx, start_idx + self.input_frames * self.n_frames_interval, self.n_frames_interval)
        output_indices = np.arange(
            start_idx + self.input_frames * self.n_frames_interval,
            start_idx + (self.input_frames + self.output_frames) * self.n_frames_interval,
            self.n_frames_interval,
        )
        all_indices = np.concatenate([input_indices, output_indices])

        model_info = {}
        model_info["model"] = model_name
        model_info["indices"] = all_indices
        
        model_data = {}
        model_data['model'] = model_name
        model_data['start_idx'] = torch.tensor(start_idx).long()
        
        model_metas = h5py.File(os.path.join(self.dataset_path, f'{model_name}'), 'r')
        model_pcls = torch.from_numpy(np.array(model_metas['x']))

        if all_indices[-1] >= model_pcls.shape[0]:
            raise IndexError(f"Invalid frame indices {all_indices} for sequence length {model_pcls.shape[0]} in {model_name}.")

        # if model_pcls[0].shape[0] > self.pc_size:
        #     ind = np.random.default_rng(seed=self.seed).choice(model_pcls[0].shape[0], self.pc_size, replace=False)
        #     points_src = model_pcls[:1]
        #     points_tgt = model_pcls[1:(self.n_training_frames*self.n_frames_interval+1):self.n_frames_interval][:, ind]
        # else: # No need to do fps in new dataset case (input is 2048 points)
        if model_pcls[0].shape[0] > self.pc_size:
            ind = np.random.default_rng(seed=self.seed).choice(model_pcls[0].shape[0], self.pc_size, replace=False)
        else:
            ind = np.arange(model_pcls[0].shape[0])

        model_data['point_indices'] = torch.from_numpy(np.array(ind)).long()
        points_rest = model_pcls[0]
        points_src = model_pcls[input_indices]
        points_tgt = model_pcls[output_indices]

        # 1b: GT for the extra rollout chunks (chunks 1..K-1); chunk 0 stays in points_tgt above.
        # F/C/vol and all existing losses remain tied to chunk 0 only.
        points_tgt_roll = None
        if self.rollout_unroll_steps > 1:
            roll_indices = np.arange(
                start_idx + (self.input_frames + self.output_frames) * self.n_frames_interval,
                start_idx + (self.input_frames + self.output_frames * self.rollout_unroll_steps) * self.n_frames_interval,
                self.n_frames_interval,
            )
            points_tgt_roll = model_pcls[roll_indices]

        # Per-particle velocity at start_idx.
        # Rule: first frame velocity is zero; otherwise use central difference from neighboring frames.
        if start_idx == 0:
            start_vel = torch.zeros_like(model_pcls[0])
        elif self.causal_start_vel:
            # input=1 消融:后向差分,只用过去帧,避免中心差分偷看目标帧 x[start_idx+1]
            # (input=1 时 output_indices=[start_idx+1] = 目标)。scale = 每帧位移,与 eval 跨步差分一致。
            start_vel = (model_pcls[start_idx] - model_pcls[max(start_idx - 1, 0)]).float()
        else:
            prev_idx = max(start_idx - 1, 0)
            next_idx = min(start_idx + 1, model_pcls.shape[0] - 1)
            denom = max(next_idx - prev_idx, 1)
            start_vel = (model_pcls[next_idx] - model_pcls[prev_idx]) / float(denom)

        if not 'drag_point' in model_metas: # Assume drag direction cross the sphere center
            drag_dir = np.array(model_metas['drag_force'])
            drag_dir = drag_dir / np.linalg.norm(drag_dir)
            drag_point = np.array([self.cfg.norm_fac, self.cfg.norm_fac, self.cfg.norm_fac]) + drag_dir
        else:
            drag_point = np.array(model_metas['drag_point'])

        if not 'floor_height' in model_metas:
            model_data['floor_height'] = torch.from_numpy(np.array(-2.4)).unsqueeze(-1).float()
        else:
            model_data['floor_height'] = (torch.from_numpy(np.array(model_metas['floor_height'])).unsqueeze(-1).float() - self.cfg.norm_fac) / 2
        model_data['drag_point'] = (torch.from_numpy(drag_point).float() - self.cfg.norm_fac) / 2
        model_data['points_rest'] = (points_rest.float() - self.cfg.norm_fac) / 2
        model_data['points_src'] = (points_src.float() - self.cfg.norm_fac) / 2
        model_data['points_tgt'] = (points_tgt.float() - self.cfg.norm_fac) / 2
        if points_tgt_roll is not None:
            model_data['points_tgt_roll'] = (points_tgt_roll.float() - self.cfg.norm_fac) / 2
        model_data['start_vel'] = start_vel.float() / 2

        model_data['vol'] = torch.from_numpy(np.array(model_metas['vol']))
        model_data['F'] = torch.from_numpy(np.array(model_metas['F']))
        if model_data['F'].shape[0] == model_pcls.shape[0]:
            model_data['F'] = model_data['F'][output_indices]
        else:
            model_data['F'] = model_data['F'][np.clip(output_indices - 1, 0, model_data['F'].shape[0] - 1)]
        model_data['C'] = torch.from_numpy(np.array(model_metas['C']))
        if model_data['C'].shape[0] == model_pcls.shape[0]:
            model_data['C'] = model_data['C'][output_indices]
        else:
            model_data['C'] = model_data['C'][np.clip(output_indices - 1, 0, model_data['C'].shape[0] - 1)]

        # single-frame boundary loss_F(output=1):需 input 末帧的 GT F/C 作推进起点
        # (batch['F']/['C'] 已是 output/pred 帧 = target/起点对的下一帧)。site 数据 F/C 全帧 →
        # input_indices[-1] 必有效;仅 output=1 时输出,其它臂 batch 不变。
        if self.output_frames == 1:
            F_all = torch.from_numpy(np.array(model_metas['F']))
            C_all = torch.from_numpy(np.array(model_metas['C']))
            last_in = int(input_indices[-1])
            model_data['F_src_last'] = F_all[last_in] if F_all.shape[0] == model_pcls.shape[0] \
                else F_all[np.clip(last_in - 1, 0, F_all.shape[0] - 1)]
            model_data['C_src_last'] = C_all[last_in] if C_all.shape[0] == model_pcls.shape[0] \
                else C_all[np.clip(last_in - 1, 0, C_all.shape[0] - 1)]

        mask = torch.from_numpy(np.array(model_metas['drag_mask'])).bool()

        if 'gravity' in model_metas:
            model_data['gravity'] = torch.from_numpy(np.array(model_metas['gravity'])).long().unsqueeze(0)
        else:
            # print('no gravity in model_metas')
            model_data['gravity'] = torch.from_numpy(np.array(0)).long().unsqueeze(0)

        model_data['drag_point'] = (torch.from_numpy(drag_point).float() - self.cfg.norm_fac) / 2
        if model_data['drag_point'].ndim == 1: # For compatibility: only have one force
            model_data['drag_point'] = torch.cat([model_data['drag_point'], torch.tensor([mask.sum()]).float()], dim=0).unsqueeze(0)
        else:
            model_data['drag_point'] = torch.cat([model_data['drag_point'], mask.sum(dim=-1, keepdim=True).float()], dim=1)

        force_order = torch.randperm(self.max_num_forces) if self.split == 'train' else torch.arange(self.max_num_forces)
        mask = mask.unsqueeze(0) if mask.ndim == 1 else mask
        # force_mask = torch.ones(self.max_num_forces, 1)
        # force_mask[:mask.shape[0]] *= 0
        # force_mask = force_mask[force_order].bool()

        if mask.shape[1] == 0:
            mask = torch.zeros(0, self.pc_size).bool()
            model_data['force'] = torch.zeros(0, 3)
            model_data['drag_point'] = torch.zeros(0, 4)
            model_data['base_drag_coeff'] = torch.zeros(self.max_num_forces, 1)
        elif not 'base_drag_coeff' in model_metas:
            vol = model_data['vol'].unsqueeze(0)
            total_volume = torch.sum(vol)
            masked_volume = torch.sum(vol * mask, dim=1)
            mean_masked_volume = masked_volume / mask.sum(dim=1)
            mask_ratio = masked_volume / total_volume
            base_drag_coeff = 9.8 * 1000 * mean_masked_volume / mask_ratio
            weighted_force = torch.from_numpy(np.array(model_metas['drag_force'])).float()
            weighted_force = weighted_force.unsqueeze(0) if weighted_force.ndim == 1 else weighted_force
            model_data['force'] = weighted_force / base_drag_coeff.unsqueeze(1)
            coeff = torch.zeros(self.max_num_forces, 1)
            coeff = coeff[force_order]
            coeff[:base_drag_coeff.shape[0]] = base_drag_coeff.unsqueeze(1)
            model_data['base_drag_coeff'] = coeff
            # model_data['weighted_force'] = weighted_force
        else:
            model_data['force'] = torch.from_numpy(np.array(model_metas['drag_force'])).float()
            model_data['base_drag_coeff'] = torch.from_numpy(np.array(model_metas['base_drag_coeff'])).float()
        
        model_data['is_mpm'] = torch.tensor(1).bool()
        if 'mat_type' in model_metas:
            model_data['mat_type'] = torch.from_numpy(np.array(model_metas['mat_type'])).long()
            if np.array(model_data['mat_type']).item() == 3: # Rigid dataset
                model_data['is_mpm'] = torch.tensor(0).bool()
        else: # temporary fix for elastic data
            model_data['mat_type'] = torch.tensor(0).long()
        
        if self.has_gravity and model_data['gravity'][0] == 1: # add gravity to force
            gravity = torch.tensor([[0, -1.0, 0]]).float()
            drag_point = (model_data['points_src'][0] * (model_data['vol'] / model_data['vol'].sum()).unsqueeze(1)).sum(axis=0) if model_data['is_mpm'] else model_data['points_src'][0].mean(axis=0)
            drag_point = torch.cat([drag_point, torch.tensor([self.pc_size]).float()]).unsqueeze(0)
            assert model_data['force'].sum() == 0, f'we are not supporting both drag and gravity now: {model_name}'
            model_data['force'] = torch.cat([model_data['force'], gravity], dim=0) if not model_data['force'].sum() == 0 else gravity
            model_data['drag_point'] = torch.cat([model_data['drag_point'], drag_point], dim=0) if not drag_point.sum() == 0 else drag_point
            mask = torch.cat([mask, torch.ones_like(mask).bool()], dim=0) if not mask.sum() == 0 else torch.ones(1, self.pc_size).bool()
        
        all_forces = torch.zeros(self.max_num_forces, 3)
        all_forces[:model_data['force'].shape[0]] = model_data['force']
        all_forces = all_forces[force_order]
        model_data['force'] = all_forces

        all_drag_points = torch.zeros(self.max_num_forces, 4)
        all_drag_points[:model_data['drag_point'].shape[0]] = model_data['drag_point']
        all_drag_points = all_drag_points[force_order]
        model_data['drag_point'] = all_drag_points

        if model_pcls[0].shape[0] > self.pc_size:
            model_data['points_rest'] = model_data['points_rest'][ind]
            model_data['points_src'] = model_data['points_src'][:, ind]
            model_data['points_tgt'] = model_data['points_tgt'][:, ind]
            if 'points_tgt_roll' in model_data:
                model_data['points_tgt_roll'] = model_data['points_tgt_roll'][:, ind]
            model_data['start_vel'] = model_data['start_vel'][ind]
            mask = mask[:, ind] if mask.shape[-1] > self.pc_size else mask

        all_mask = torch.zeros(self.max_num_forces, self.pc_size).bool()
        all_mask[:mask.shape[0]] = mask
        all_mask = all_mask[force_order]

        model_data['mask'] = all_mask[..., None] # (n_forces, pc_size, 1) for compatibility
        model_data['E'] = torch.log10(torch.from_numpy(np.array(model_metas['E'])).unsqueeze(-1).float()) if np.array(model_metas['E']) > 0 else torch.zeros(1).float()
        model_data['nu'] = torch.from_numpy(np.array(model_metas['nu'])).unsqueeze(-1).float()

        return model_data, model_info
        model_data['force'] = all_forces

        all_drag_points = torch.zeros(self.max_num_forces, 4)
        all_drag_points[:model_data['drag_point'].shape[0]] = model_data['drag_point']
        all_drag_points = all_drag_points[force_order]
        model_data['drag_point'] = all_drag_points

        if model_pcls[0].shape[0] > self.pc_size:
            model_data['points_src'] = model_data['points_src'][:, ind]
            model_data['points_tgt'] = model_data['points_tgt'][:, ind]
            if 'points_tgt_roll' in model_data:
                model_data['points_tgt_roll'] = model_data['points_tgt_roll'][:, ind]
            model_data['start_vel'] = model_data['start_vel'][ind]
            mask = mask[:, ind] if mask.shape[-1] > self.pc_size else mask

        all_mask = torch.zeros(self.max_num_forces, self.pc_size).bool()
        all_mask[:mask.shape[0]] = mask
        all_mask = all_mask[force_order]

        model_data['mask'] = all_mask[..., None] # (n_forces, pc_size, 1) for compatibility
        model_data['E'] = torch.log10(torch.from_numpy(np.array(model_metas['E'])).unsqueeze(-1).float()) if np.array(model_metas['E']) > 0 else torch.zeros(1).float()
        model_data['nu'] = torch.from_numpy(np.array(model_metas['nu'])).unsqueeze(-1).float()

        return model_data, model_info
