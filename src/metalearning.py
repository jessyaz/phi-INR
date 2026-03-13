import torch
import torch.nn as nn

# ── Inner loop ───────────────────────────────────────────────

def inner_loop(func_rep, modulations, coords, features, x_statics, y_target, inner_steps, inner_lr):
    for _ in range(inner_steps):
        modulations = inner_loop_step(
            func_rep, modulations, coords, features, x_statics, y_target, inner_lr
        )
    return modulations


def inner_loop_step(func_rep, modulations, coords, features, x_statics, y_target, inner_lr):
    loss_fn = nn.MSELoss(reduction="mean")

    with torch.enable_grad():
        y_hat, _ = func_rep.modulated_forward(coords, modulations, features, x_statics)
        loss     = loss_fn(y_hat, y_target)
        grad     = torch.autograd.grad(loss, modulations, create_graph=False)[0]

    return modulations - inner_lr * grad


# ── Outer step ───────────────────────────────────────────────

def outer_step(
        func_rep,
        coords_p,
        coords_h,
        features_p,
        x_statics,
        y_target_p,
        y_target_h,
        inner_steps,
        inner_lr,
        w_passed,
        w_futur,
        is_train=False,
        modulations=0,
        beta=0,
):
    loss_fn = nn.MSELoss(reduction="none")

    func_rep.zero_grad()
    modulations = modulations.requires_grad_()

    modulations = inner_loop(
        func_rep, modulations, coords_p, features_p,
        x_statics, y_target_p, inner_steps, inner_lr,
    )

    with torch.set_grad_enabled(is_train):
        y_hat_p, _ = func_rep.modulated_forward(coords_p, modulations, features_p, x_statics, beta=beta)
        y_hat_h, _ = func_rep.modulated_forward(coords_h, modulations, features_p, x_statics, beta=beta)

        loss = (w_passed * loss_fn(y_hat_p, y_target_p) + w_futur * loss_fn(y_hat_h, y_target_h)).mean()

    return {
        "loss":        loss,
        "modulations": modulations,
    }