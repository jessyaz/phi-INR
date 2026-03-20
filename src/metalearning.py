import torch
import torch.nn as nn


def inner_loop_step(func_rep, code, coords, hs_p, x_statics, dir_idx, y_past, inner_lr):
    loss_fn = nn.MSELoss()
    with torch.enable_grad():
        code  = code.detach().requires_grad_(True)
        out_p = func_rep._inr_batch(coords, code, hs_p, x_statics, dir_idx) \
            if func_rep.use_context \
            else func_rep._inr_batch_vanilla(coords, code)
        loss = loss_fn(out_p, y_past)
        grad = torch.autograd.grad(loss, code, create_graph=False)[0]
    return (code - inner_lr * grad).detach()


def inner_loop(func_rep, code, coords, hs_p, x_statics, dir_idx, y_past,
               inner_steps, inner_lr):
    for _ in range(inner_steps):
        code = inner_loop_step(
            func_rep, code, coords, hs_p, x_statics, dir_idx, y_past, inner_lr
        )
    return code


def outer_step(
        func_rep,
        coords_p, coords_h,
        x_context_p, x_context_h,
        y_past, y_horizon,
        inner_steps, inner_lr,
        w_passed, w_futur,
        is_train  = False,
        code      = None,
        x_statics = None,
        dir_idx   = None,
        beta      = 0,
):
    loss_fn = nn.MSELoss()
    func_rep.zero_grad()

    B = coords_p.shape[0]
    if code is None:
        code = torch.zeros(B, func_rep.latent_dim, device=coords_p.device)
    code = code.detach().requires_grad_(True)

    # ── Précalcul hs_p une seule fois (LSTM fixe pendant inner loop) ──
    if func_rep.use_context:
        with torch.no_grad():
            hs_p, h_final, c_final = func_rep.lstm.forward_past(x_context_p, y_past)
    else:
        hs_p, h_final, c_final = None, None, None

    # ── Inner loop — seul code évolue ────────────────────────
    code = inner_loop(
        func_rep, code, coords_p, hs_p,
        x_statics, dir_idx, y_past,
        inner_steps, inner_lr,
    )

    # ── Outer forward complet ─────────────────────────────────
    with torch.set_grad_enabled(is_train):

        if func_rep.use_context:
            # Recalcul avec grad pour l'outer loss
            hs_p, h_final, c_final = func_rep.lstm.forward_past(x_context_p, y_past)
            out_p = func_rep._inr_batch(coords_p, code, hs_p, x_statics, dir_idx)

            out_h = None
            if coords_h is not None: # Toujours autoregressif ici
                B         = coords_h.shape[0]
                T_h       = coords_h.shape[1]
                h, c      = h_final, c_final
                out_h     = out_p[:, -1, :]
                hs_h      = torch.zeros(B, T_h ,h.shape[-1], device = coords_h.device)
                flow_h    = torch.zeros(B, T_h, device = coords_h.device)
                for t in range(T_h):

                    h, c    = func_rep.lstm.cell_step( torch.cat([x_context_h[:, t, :], out_h] , dim=-1), h, c)

                    out_h = func_rep._inr_batch(coords_h[:,t].unsqueeze(-1), code, h.unsqueeze(1), x_statics, dir_idx).squeeze(1)
                    hs_h[:,t,:] = h
                    flow_h[:,t] = out_h.squeeze(-1)

               # hs_h  = torch.cat(hs_h, dim=1)
               # out_h = func_rep._inr_batch(coords_h, code, hs_h, x_statics, dir_idx)
        else:
            out_p = func_rep._inr_batch_vanilla(coords_p, code)
            out_h = func_rep._inr_batch_vanilla(coords_h, code) #if coords_h is not None else None


        loss_p = loss_fn(out_p, y_past)

        loss_h = loss_fn(flow_h.unsqueeze(-1), y_horizon) #if out_h is not None else torch.tensor(0.0, device=coords_p.device)
        loss   = w_passed * loss_p + w_futur * loss_h

    return {
        'loss':   loss,
        'loss_p': loss_p,
        'loss_h': loss_h,
        'code':   code.detach(),
        'out_p':  out_p.detach(),
        'out_h':  out_h.detach() #if out_h is not None else None,
    }
