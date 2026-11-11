import torch
from infer.local import build_model, compile_aoti


def _save_exported(model, example, path):
    # batch left free
    exported = torch.export.export(
        model,
        example,
        dynamic_shapes=tuple({0: torch.export.Dim.AUTO} for _ in example),
    )
    torch.export.save(exported, path)


class TwoInputs(torch.nn.Module):
    def forward(self, x, y):
        return x.sum((1, 2)) + y.sum((1, 2))


def test_compiled_package_runs_any_batch_size(tmp_path):
    model = torch.nn.Conv1d(2, 4, 3)
    _save_exported(model, (torch.randn(8, 2, 16),), tmp_path / "weights.pt2")

    compile_aoti(tmp_path / "weights.pt2", tmp_path / "package.pt2", "cpu")

    compiled = build_model(None, "aoti", "cpu", tmp_path / "package.pt2")
    x = torch.randn(3, 2, 16)
    torch.testing.assert_close(compiled(x), model(x))


def test_compiled_package_runs_a_model_with_two_inputs(tmp_path):
    model = TwoInputs()
    example = (torch.randn(8, 2, 16), torch.randn(8, 6, 4))
    _save_exported(model, example, tmp_path / "weights.pt2")

    compile_aoti(tmp_path / "weights.pt2", tmp_path / "package.pt2", "cpu")

    compiled = build_model(None, "aoti", "cpu", tmp_path / "package.pt2")
    x, y = torch.randn(3, 2, 16), torch.randn(3, 6, 4)
    torch.testing.assert_close(compiled(x, y), model(x, y))
