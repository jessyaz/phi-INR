import torch
import torch.nn as nn

def inner_loop_step(func_rep, code, coords, hs_p, x_statics, dir_idx, y_past, inner_lr, is_train):
    loss_fn = nn.MSELoss()

    with torch.enable_grad():
        if func_rep.use_context and func_rep.control != "static_only":
            out_p = func_rep._inr_batch(coords, code, hs_p, x_statics, dir_idx)
        elif func_rep.use_context and func_rep.control == "static_only":
            out_p = func_rep._inr_batch_static_only(coords, code, x_statics, dir_idx)
        else:
            out_p = func_rep._inr_batch_vanilla(coords, code)

        loss = loss_fn(out_p, y_past)
        grad = torch.autograd.grad(loss, code, create_graph=is_train)[0]

    new_code = code - inner_lr * grad

    if is_train:
        return new_code
    else:
        return new_code.detach().requires_grad_(True)


def inner_loop(func_rep, code, coords, hs_p, x_statics, dir_idx, y_past, inner_steps, inner_lr, is_train=False):
    for _ in range(inner_steps):
        code = inner_loop_step(func_rep, code, coords, hs_p, x_statics, dir_idx, y_past, inner_lr, is_train)
    return code


def outer_step(func_rep, coords_p, coords_h, x_context_p, x_context_h, y_past, y_horizon, inner_steps, inner_lr, w_passed, w_futur, is_train=False, code=None, x_statics=None, dir_idx=None, beta=0):
    loss_fn = nn.MSELoss()
    func_rep.zero_grad()

    B = coords_p.shape[0]
    if code is None:
        code = torch.zeros(B, func_rep.latent_dim, device=coords_p.device)

    code = code.detach().requires_grad_(True)

    if func_rep.use_context and func_rep.control != "static_only":
        with torch.no_grad():
            hs_p, h_final, c_final = func_rep.lstm.forward_past(x_context_p, y_past)
    else:
        hs_p, h_final, c_final = None, None, None

    code = inner_loop(func_rep, code, coords_p, hs_p, x_statics, dir_idx, y_past, inner_steps, inner_lr, is_train)

    with torch.set_grad_enabled(is_train):
        B = coords_h.shape[0]
        T_h = coords_h.shape[1]
        flow_h = torch.zeros(B, T_h, device=coords_h.device)

        if func_rep.use_context and func_rep.control != "static_only":
            hs_p, h_final, c_final = func_rep.lstm.forward_past(x_context_p, y_past)
            out_p = func_rep._inr_batch(coords_p, code, hs_p, x_statics, dir_idx)

            h, c = h_final, c_final
            out_h = out_p[:, -1, :]
            hs_h = torch.zeros(B, T_h, h.shape[-1], device=coords_h.device)

            for t in range(T_h):
                h, c = func_rep.lstm.cell_step(torch.cat([x_context_h[:, t, :], out_h], dim=-1), h, c)
                out_h = func_rep._inr_batch(coords_h[:, t].unsqueeze(-1), code, h.unsqueeze(1), x_statics, dir_idx).squeeze(1)
                hs_h[:, t, :] = h
                flow_h[:, t] = out_h.squeeze(-1)
            flow_h = flow_h.unsqueeze(-1)

        elif func_rep.use_context and func_rep.control == "static_only":
            out_p = func_rep._inr_batch_static_only(coords_p, code, x_statics, dir_idx)
            flow_h = func_rep._inr_batch_static_only(coords_h, code, x_statics, dir_idx)

        else:
            out_p = func_rep._inr_batch_vanilla(coords_p, code)
            flow_h = func_rep._inr_batch_vanilla(coords_h, code)

        loss_p = loss_fn(out_p, y_past)
        loss_h = loss_fn(flow_h, y_horizon)
        loss = w_passed * loss_p + w_futur * loss_h

    return {
        'loss': loss,
        'loss_p': loss_p,
        'loss_h': loss_h,
        'code': code.detach(),
        'out_p': out_p.detach(),
        'out_h': flow_h.detach()
    }