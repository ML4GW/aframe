import torch
from infer.local import build_model, compile_aoti


def test_compiled_package_runs_any_batch_size(tmp_path):
    model = torch.nn.Conv1d(2, 4, 3)
    # batch left free, as in model_exported.pt2
    exported = torch.export.export(
        model,
        (torch.randn(8, 2, 16),),
        dynamic_shapes=({0: torch.export.Dim.AUTO},),
    )
    torch.export.save(exported, tmp_path / "weights.pt2")

    compile_aoti(
        tmp_path / "weights.pt2", tmp_path / "package.pt2", 8, 2, 16, "cpu"
    )

    compiled = build_model(None, "aoti", "cpu", tmp_path / "package.pt2")
    x = torch.randn(3, 2, 16)
    torch.testing.assert_close(compiled(x), model(x))
