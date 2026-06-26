import torch
from diffusers import DiffusionPipeline


class TrajPipeline(DiffusionPipeline):
    def __init__(self, model, scheduler=None):
        super().__init__()
        if scheduler is None:
            self.register_modules(model=model)
            self.scheduler = None
        else:
            self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(self, init_pc, force, E, nu, mask, drag_point, floor_height, gravity, coeff,
        generator, 
        device, 
        y = None,
        start_vel = None,
        points_rest = None,
        batch_size: int = 1, 
        num_inference_steps: int = 50, 
        guidance_scale=1.0, 
        n_frames=20
    ):
        if self.scheduler is not None:
            sample = torch.randn((batch_size, n_frames, init_pc.shape[2], 3), generator=generator).to(device)
        else:
            # Use the last source frame repeated as input, instead of zeros,
            # so that PointEmbed produces differentiated per-point embeddings.
            # Add small noise to break uniformity across frames.
            sample = init_pc[:, -1:, :, :].repeat(1, n_frames, 1, 1).to(device)
            # generator-controlled noise so eval is reproducible. torch.randn_like ignores
            # `generator` -> would draw from the uncontrolled global RNG (eval sets no global
            # seed). Matters more at output_frames=1 (noise drawn ~20x per rollout vs ~4x).
            # Mirror the scheduler branch above: draw on CPU with the seeded gen, then .to(device).
            noise = torch.randn(sample.shape, generator=generator, dtype=sample.dtype) * 0.02
            sample = sample + noise.to(device)
        self.model.to(device)
        init_pc = init_pc.to(device)
        force = force.to(device)
        E = E.to(device)
        nu = nu.to(device)
        mask = mask.to(device).to(dtype=sample.dtype)
        drag_point = drag_point.to(device)
        floor_height = floor_height.to(device)
        coeff = coeff.to(device)
        start_vel = start_vel.to(device) if start_vel is not None else None
        points_rest = points_rest.to(device) if points_rest is not None else init_pc[:, 0]
        gravity = gravity.to(device) if gravity is not None else None
        y = y.to(device) if y is not None else None

        do_classifier_free_guidance = (guidance_scale > 1.0)
        null_emb = torch.tensor([1] * batch_size).to(sample.dtype)
        if do_classifier_free_guidance:
            init_pc = torch.cat([init_pc] * 2)
            force = torch.cat([force] * 2)
            E = torch.cat([E] * 2)
            nu = torch.cat([nu] * 2)
            mask = torch.cat([mask] * 2)
            drag_point = torch.cat([drag_point] * 2)
            floor_height = torch.cat([floor_height] * 2)
            points_rest = torch.cat([points_rest] * 2)
            if start_vel is not None:
                start_vel = torch.cat([start_vel] * 2)
            null_emb = torch.cat([torch.tensor([0] * batch_size).to(sample.dtype), null_emb])
        null_emb = null_emb[:, None, None].to(device)
        if self.scheduler is None:
            t = torch.zeros((batch_size,), device=device, dtype=torch.long)
            sample_input = torch.cat([sample] * 2) if do_classifier_free_guidance else sample
            t = torch.cat([t] * 2) if do_classifier_free_guidance else t
            model_output = self.model(sample_input, t, init_pc, force, E, nu, mask, drag_point, floor_height=floor_height, gravity_label=gravity, coeff=coeff, y=y, null_emb=null_emb, start_vel=start_vel, points_rest=points_rest)
            if do_classifier_free_guidance:
                model_pred_uncond, model_pred_cond = model_output.chunk(2)
                model_output = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)
            sample = model_output
        else:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            for t in self.progress_bar(self.scheduler.timesteps):
                t = torch.tensor([t] * batch_size, device=device)
                sample_input = torch.cat([sample] * 2) if do_classifier_free_guidance else sample
                t = torch.cat([t] * 2) if do_classifier_free_guidance else t
                model_output = self.model(sample_input, t, init_pc, force, E, nu, mask, drag_point, floor_height=floor_height, gravity_label=gravity, coeff=coeff, y=y, null_emb=null_emb, start_vel=start_vel, points_rest=points_rest)
                if do_classifier_free_guidance:
                    model_pred_uncond, model_pred_cond = model_output.chunk(2)
                    model_output = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)
                sample = self.scheduler.step(model_output, t[0], sample).prev_sample
        return sample
