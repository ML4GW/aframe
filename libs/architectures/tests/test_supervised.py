import tempfile

import h5py
import numpy as np
import torch
from architectures.supervised import (
    SupervisedDecimatedResNet,
    SupervisedDecimatedSVDNet,
)


class TestSupervisedDecimatedResNet:
    def test_forward_shape(self):
        """4 branches with shared config produce scalar output."""
        batch = 8
        num_ifos = 2
        num_branches = 4
        arch = SupervisedDecimatedResNet(
            num_ifos=num_ifos,
            num_branches=num_branches,
            branch_layers=[2, 2],
            branch_classes=16,
        )
        segments = [torch.randn(batch, num_ifos, 128) for _ in range(4)]
        out = arch(*segments)
        assert out.shape == (batch,)

    def test_per_branch_classes(self):
        """Per-branch embedding dims via list of ints."""
        batch = 4
        num_ifos = 2
        arch = SupervisedDecimatedResNet(
            num_ifos=num_ifos,
            num_branches=3,
            branch_layers=[2, 2],
            branch_classes=[8, 16, 32],
        )
        segments = [torch.randn(batch, num_ifos, 64) for _ in range(3)]
        out = arch(*segments)
        assert out.shape == (batch,)

    def test_per_branch_layers(self):
        """Per-branch layer configs via list of lists."""
        batch = 4
        num_ifos = 2
        arch = SupervisedDecimatedResNet(
            num_ifos=num_ifos,
            num_branches=2,
            branch_layers=[[2, 2], [3, 3]],
            branch_classes=16,
        )
        segments = [torch.randn(batch, num_ifos, 128) for _ in range(2)]
        out = arch(*segments)
        assert out.shape == (batch,)


class TestSupervisedDecimatedSVDNet:
    """Tests for frequency-domain SVD network.

    Without an SVD basis file, n_freq falls back to n_svd[i],
    so input segments must have n_samples = (n_svd - 1) * 2.
    """

    def _n_samples(self, n_svd):
        """Compute required n_samples for a given n_svd."""
        return (n_svd - 1) * 2

    def test_shallow_forward(self):
        """Shallow network (no SVD basis file) produces scalar output."""
        batch = 8
        num_ifos = 2
        n_svd = 16
        n_samples = self._n_samples(n_svd)
        arch = SupervisedDecimatedSVDNet(
            num_ifos=num_ifos,
            num_branches=2,
            n_svd=n_svd,
            branch_hidden_dims="shallow",
            branch_embed_dim=8,
            freeze_svd=False,
        )
        segments = [
            torch.randn(batch, num_ifos, n_samples) for _ in range(2)
        ]
        out = arch(*segments)
        assert out.shape == (batch,)

    def test_flat_forward(self):
        """Flat (legacy) dense network produces scalar output."""
        batch = 4
        num_ifos = 2
        n_svd = 16
        n_samples = self._n_samples(n_svd)
        arch = SupervisedDecimatedSVDNet(
            num_ifos=num_ifos,
            num_branches=2,
            n_svd=n_svd,
            branch_hidden_dim=32,
            num_dense_blocks=2,
            branch_embed_dim=8,
            freeze_svd=False,
        )
        segments = [
            torch.randn(batch, num_ifos, n_samples) for _ in range(2)
        ]
        out = arch(*segments)
        assert out.shape == (batch,)

    def test_tapering_forward(self):
        """Tapering dense network produces scalar output."""
        batch = 4
        num_ifos = 2
        n_svd = 16
        n_samples = self._n_samples(n_svd)
        arch = SupervisedDecimatedSVDNet(
            num_ifos=num_ifos,
            num_branches=2,
            n_svd=n_svd,
            branch_hidden_dims=[64, 32],
            branch_embed_dim=8,
            num_blocks_per_stage=1,
            freeze_svd=False,
        )
        segments = [
            torch.randn(batch, num_ifos, n_samples) for _ in range(2)
        ]
        out = arch(*segments)
        assert out.shape == (batch,)

    def test_freeze_unfreeze(self):
        """set_svd_frozen toggles requires_grad on SVD params."""
        arch = SupervisedDecimatedSVDNet(
            num_ifos=2,
            num_branches=2,
            n_svd=16,
            branch_hidden_dims="shallow",
            branch_embed_dim=8,
            freeze_svd=True,
        )
        # Initially frozen
        for svd_layer in arch.svd_layers:
            for p in svd_layer.parameters():
                assert not p.requires_grad

        # Unfreeze
        arch.set_svd_frozen(False)
        for svd_layer in arch.svd_layers:
            for p in svd_layer.parameters():
                assert p.requires_grad

        # Re-freeze
        arch.set_svd_frozen(True)
        for svd_layer in arch.svd_layers:
            for p in svd_layer.parameters():
                assert not p.requires_grad

    def test_normalize_svd(self):
        """normalize_svd=True still produces correct output shape."""
        batch = 4
        num_ifos = 2
        n_svd = 16
        n_samples = self._n_samples(n_svd)
        arch = SupervisedDecimatedSVDNet(
            num_ifos=num_ifos,
            num_branches=2,
            n_svd=n_svd,
            branch_hidden_dims="shallow",
            branch_embed_dim=8,
            normalize_svd=True,
            freeze_svd=False,
        )
        segments = [
            torch.randn(batch, num_ifos, n_samples) for _ in range(2)
        ]
        out = arch(*segments)
        assert out.shape == (batch,)
        # Check that LayerNorm was used (not Identity)
        for norm in arch.svd_norms:
            assert isinstance(norm, torch.nn.LayerNorm)

    def test_per_ifo_svd(self):
        """per_ifo_svd=True forward pass produces correct shape."""
        batch = 4
        num_ifos = 2
        n_svd = 16
        n_samples = self._n_samples(n_svd)
        arch = SupervisedDecimatedSVDNet(
            num_ifos=num_ifos,
            num_branches=2,
            n_svd=n_svd,
            branch_hidden_dims="shallow",
            branch_embed_dim=8,
            per_ifo_svd=True,
            freeze_svd=False,
        )
        segments = [
            torch.randn(batch, num_ifos, n_samples) for _ in range(2)
        ]
        out = arch(*segments)
        assert out.shape == (batch,)

    def test_load_from_hdf5(self):
        """Loading V matrices from HDF5 produces correct shape."""
        batch = 4
        num_ifos = 2
        n_svd = 8
        n_freq = 33  # e.g. from a 64-sample segment
        n_samples = (n_freq - 1) * 2  # 64

        with tempfile.NamedTemporaryFile(suffix=".hdf5") as tmp:
            # Create HDF5 with V matrices
            with h5py.File(tmp.name, "w") as f:
                for i in range(2):
                    grp = f.create_group(f"branch_{i}")
                    grp.create_dataset(
                        "V", data=np.random.randn(2 * n_freq, n_svd)
                    )
                    grp.attrs["n_freq"] = n_freq

            arch = SupervisedDecimatedSVDNet(
                num_ifos=num_ifos,
                num_branches=2,
                n_svd=n_svd,
                branch_hidden_dims="shallow",
                branch_embed_dim=8,
                svd_basis_path=tmp.name,
                freeze_svd=False,
            )
            segments = [
                torch.randn(batch, num_ifos, n_samples) for _ in range(2)
            ]
            out = arch(*segments)
            assert out.shape == (batch,)
