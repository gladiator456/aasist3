"""Quick forward/backward sanity check; does not require the ASVspoof dataset."""
import torch
from MODELDEFINITION import Model

def main():
    torch.manual_seed(7)
    args = {
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
        "gat_dims": [24, 32],
        "pool_ratios": [0.4, 0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 1.0, 1.0],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(args).to(device).train()
    waves = torch.randn(1, 64600, device=device)
    labels = torch.tensor([1], device=device)
    _, logits = model(waves)
    assert logits.shape == (1, 2), f"Unexpected logits shape: {tuple(logits.shape)}"
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    assert torch.isfinite(loss), f"Non-finite loss: {loss.item()}"
    grad_count = sum(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert grad_count > 0, "No gradients were produced"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), "Non-finite gradients found"
    # Multi-sample batch in eval mode (what validation does), and a check that
    # repeated inference is deterministic.
    model.eval()
    with torch.no_grad():
        batch = torch.randn(4, 64600, device=device)
        _, out1 = model(batch)
        _, out2 = model(batch)
    assert out1.shape == (4, 2), f"Unexpected eval logits shape: {tuple(out1.shape)}"
    assert torch.isfinite(out1).all(), "Non-finite eval logits"
    assert torch.allclose(out1, out2), "Eval mode is not deterministic"
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print(f"Smoke test passed on {device}. logits={tuple(logits.shape)}, loss={loss.item():.4f}, parameters_with_grad={grad_count}")

if __name__ == "__main__":
    main()
