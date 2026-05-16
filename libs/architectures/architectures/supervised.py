from typing import Literal, Optional

from architectures import Architecture
from architectures.networks import S4Model, WaveNet, Xylophone
from jaxtyping import Float
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D
from ml4gw.nn.resnet.resnet_2d import ResNet2D
from ml4gw.nn.svd import DenseResidualBlock, FreqDomainSVDProjection
from torch import Tensor
import torch


class SupervisedArchitecture(Architecture):
    """
    Dummy class for registering available architectures
    for supervised learning problems. Supervised architectures
    are expected to return a single, real-valued logit
    corresponding to a detection statistic.
    """

    def forward(
        self, X: Float[Tensor, "batch channels ..."]
    ) -> Float[Tensor, " batch"]:
        raise NotImplementedError


class SupervisedTimeDomainResNet(ResNet1D, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__(
            num_ifos,
            layers=layers,
            classes=1,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )


class SupervisedFrequencyDomainResNet(ResNet1D, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__(
            num_ifos * 2,
            layers=layers,
            classes=1,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )


class SupervisedTimeDomainXylophone(Xylophone, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        norm_layer: Optional[NormLayer] = None,
    ):
        super().__init__(
            num_ifos,
            classes=1,
            norm_layer=norm_layer,
        )


class SupervisedTimeDomainWaveNet(WaveNet, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        res_channels: int,
        layers_per_block: int,
        num_blocks: int,
        kernel_size: int = 2,
        norm_layer: Optional[NormLayer] = None,
    ):
        super().__init__(
            num_ifos,
            res_channels,
            layers_per_block,
            num_blocks,
            kernel_size=kernel_size,
            norm_layer=norm_layer,
        )


class SupervisedSpectrogramDomainResNet(ResNet2D, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__(
            num_ifos,
            layers=layers,
            classes=1,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )


class SupervisedS4Model(S4Model, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        d_output: int = 1,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
        prenorm: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: Optional[float] = None,
    ) -> None:
        length = int(kernel_length * sample_rate)
        super().__init__(
            length=length,
            d_input=num_ifos,
            d_output=d_output,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            prenorm=prenorm,
            dt_min=dt_min,
            dt_max=dt_max,
            lr=lr,
        )


class SupervisedMultiModalResNet(SupervisedArchitecture):
    """
    MultiModal embedding network that embeds time, frequency, and PSD data.
    We pass the data through their own ResNets defined by their layers
    and context dims, then concatenate the output embeddings.
    """

    def __init__(
        self,
        num_ifos: int,
        time_classes: int,
        freq_classes: int,
        time_layers: list[int],
        freq_layers: list[int],
        time_kernel_size: int = 3,
        freq_kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
        **kwargs,
    ):
        super().__init__()
        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=time_layers,
            classes=time_classes,
            kernel_size=time_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        self.freq_psd_resnet = ResNet1D(
            in_channels=int(num_ifos * 3),
            layers=freq_layers,
            classes=freq_classes,
            kernel_size=freq_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        self.classifier = torch.nn.Linear(time_classes + freq_classes, 1)

    def forward(self, X, X_fft):
        time_domain_output = self.time_domain_resnet(X)
        freq_domain_output = self.freq_psd_resnet(X_fft)
        concat = torch.cat([time_domain_output, freq_domain_output], dim=-1)
        return self.classifier(concat)


class SupervisedDecimatedResNet(SupervisedArchitecture):
    """
    Multi-branch ResNet1D for decimated time-domain inputs.

    Each decimation segment gets its own ResNet1D branch that
    produces an embedding. Embeddings are concatenated and passed
    through a final linear classifier. This allows each branch to
    specialize in its frequency band.

    Args:
        num_ifos:
            Number of interferometer channels.
        num_branches:
            Number of decimation segments (branches).
        branch_layers:
            ResNet layer configuration for each branch. Can be a
            single list (shared across branches) or a list of lists
            (per-branch).
        branch_classes:
            Embedding dimension for each branch. Can be a single int
            (shared) or a list of ints (per-branch).
        kernel_size:
            Convolution kernel size for ResNet blocks.
        norm_layer:
            Normalization layer getter.
    """

    def __init__(
        self,
        num_ifos: int,
        num_branches: int,
        branch_layers: list,
        branch_classes: list[int] | int,
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
        **kwargs,
    ):
        super().__init__()
        self.num_branches = num_branches

        # Normalize branch_classes to a list
        if isinstance(branch_classes, int):
            branch_classes = [branch_classes] * num_branches

        # Normalize branch_layers: if it's a flat list of ints,
        # use same layers for all branches
        if branch_layers and isinstance(branch_layers[0], int):
            branch_layers = [branch_layers] * num_branches

        self.branches = torch.nn.ModuleList()
        for i in range(num_branches):
            self.branches.append(
                ResNet1D(
                    in_channels=num_ifos,
                    layers=branch_layers[i],
                    classes=branch_classes[i],
                    kernel_size=kernel_size,
                    zero_init_residual=zero_init_residual,
                    groups=groups,
                    width_per_group=width_per_group,
                    stride_type=stride_type,
                    norm_layer=norm_layer,
                )
            )

        total_classes = sum(branch_classes)
        self.classifier = torch.nn.Linear(total_classes, 1)

    def forward(self, *segments):
        embeddings = []
        for i, seg in enumerate(segments):
            embeddings.append(self.branches[i](seg))
        concat = torch.cat(embeddings, dim=-1)
        return self.classifier(concat).squeeze(-1)


class SupervisedDecimatedSVDNet(SupervisedArchitecture):
    """
    Multi-branch frequency-domain SVD network for BNS detection.

    Each decimation branch:
    1. FFTs its time-domain segment to frequency domain
    2. Projects onto a reduced SVD basis (filtering noise orthogonal
       to the signal manifold)
    3. Processes SVD coefficients through a dense residual network

    Branch embeddings are concatenated and classified.

    This follows the approach of frequency-domain SVD projection,
    adapted for multi-rate decimated detection.

    Args:
        num_ifos: Number of interferometer channels.
        num_branches: Number of decimation segments.
        n_svd: SVD components per branch.
        branch_hidden_dim: Hidden dimension for each branch's dense
            network (legacy, single width). Ignored if
            branch_hidden_dims is provided.
        branch_hidden_dims: Tapering hidden dimensions for each
            branch's dense network. Can be a single list like
            [512, 256, 128] (shared across branches) or a list of
            lists (per-branch). If an int, wraps in [int] for
            backward compat. Use "shallow" for a minimal
            BN -> Linear -> ReLU -> Linear architecture.
        branch_embed_dim: Output embedding dimension per branch.
        num_dense_blocks: Number of dense residual blocks per branch
            (legacy, used when branch_hidden_dims is None).
        num_blocks_per_stage: Residual blocks at each width stage
            (used with branch_hidden_dims).
        norm_type: Normalization type for dense blocks, "layer" or
            "batch".
        per_ifo_svd: If True, use per-IFO SVD projection weights.
        svd_basis_path: Path to HDF5 with precomputed V matrices.
        freeze_svd: Whether to freeze SVD layers initially.
        dropout: Dropout rate in dense blocks.
        normalize_svd: If True, apply LayerNorm to SVD output
            before the dense network. Recommended for stable
            training when SVD coefficients have large scale.
    """

    def __init__(
        self,
        num_ifos: int,
        num_branches: int,
        n_svd: list[int] | int = 100,
        branch_hidden_dim: list[int] | int = 128,
        branch_hidden_dims: Optional[list | int | str] = None,
        branch_embed_dim: list[int] | int = 32,
        num_dense_blocks: int = 3,
        num_blocks_per_stage: int = 2,
        norm_type: str = "layer",
        per_ifo_svd: bool = False,
        svd_basis_path: Optional[str] = None,
        freeze_svd: bool = True,
        dropout: float = 0.1,
        normalize_svd: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_branches = num_branches

        if isinstance(n_svd, int):
            n_svd = [n_svd] * num_branches
        if isinstance(branch_embed_dim, int):
            branch_embed_dim = [branch_embed_dim] * num_branches

        # Determine dense network architecture
        use_shallow = branch_hidden_dims == "shallow"
        use_tapering = (
            branch_hidden_dims is not None and not use_shallow
        )
        if use_tapering:
            # Normalize branch_hidden_dims
            if isinstance(branch_hidden_dims, int):
                branch_hidden_dims = [branch_hidden_dims]
            # If it's a flat list of ints, share across branches
            if branch_hidden_dims and isinstance(
                branch_hidden_dims[0], int
            ):
                branch_hidden_dims = [
                    branch_hidden_dims
                ] * num_branches
        elif not use_shallow:
            # Legacy: single hidden dim with num_dense_blocks
            if isinstance(branch_hidden_dim, int):
                branch_hidden_dim = [branch_hidden_dim] * num_branches

        # Load precomputed SVD bases
        V_matrices, n_freqs = self._load_svd_bases(
            svd_basis_path, num_branches
        )

        self.svd_layers = torch.nn.ModuleList()
        self.svd_norms = torch.nn.ModuleList()
        self.branches = torch.nn.ModuleList()

        for i in range(num_branches):
            V = V_matrices[i] if V_matrices else None
            n_freq = n_freqs[i] if n_freqs else n_svd[i]

            # SVD projection: time -> freq -> n_svd coefficients per IFO
            svd_layer = FreqDomainSVDProjection(
                num_ifos, n_freq, n_svd[i], V,
                per_channel=per_ifo_svd,
            )
            if freeze_svd:
                svd_layer.freeze()
            self.svd_layers.append(svd_layer)

            # Optional normalization on SVD output
            # Use LayerNorm (not BatchNorm) to avoid train/eval
            # discrepancy where BatchNorm causes output collapse
            svd_out_dim = n_svd[i] * num_ifos
            if normalize_svd:
                self.svd_norms.append(
                    torch.nn.LayerNorm(svd_out_dim)
                )
            else:
                self.svd_norms.append(torch.nn.Identity())

            # Dense network: SVD coefficients -> embedding
            e_dim = branch_embed_dim[i]

            if use_shallow:
                layers = self._build_shallow_network(
                    svd_out_dim, e_dim, dropout,
                )
            elif use_tapering:
                dims = branch_hidden_dims[i]
                layers = self._build_tapering_network(
                    svd_out_dim, dims, e_dim,
                    num_blocks_per_stage, dropout,
                )
            else:
                h_dim = branch_hidden_dim[i]
                layers = self._build_flat_network(
                    svd_out_dim, h_dim, e_dim,
                    num_dense_blocks, dropout,
                )

            self.branches.append(torch.nn.Sequential(*layers))

        total_embed = sum(branch_embed_dim)
        self.classifier = torch.nn.Linear(total_embed, 1)

    @staticmethod
    def _build_shallow_network(in_dim, out_dim, dropout):
        """Build minimal dense network: Linear -> ReLU -> Linear.

        Designed for weak-signal detection where a simpler model
        generalizes better than a deep one.
        """
        hidden = min(in_dim, 128)
        return [
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, out_dim),
        ]

    @staticmethod
    def _build_flat_network(
        in_dim, hidden_dim, out_dim, num_blocks, dropout,
    ):
        """Build legacy flat dense network (single hidden width)."""
        layers = [
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.GELU(),
        ]
        for _ in range(num_blocks):
            layers.append(
                DenseResidualBlock(hidden_dim, dropout)
            )
        layers.append(torch.nn.Linear(hidden_dim, out_dim))
        return layers

    @staticmethod
    def _build_tapering_network(
        in_dim, hidden_dims, out_dim,
        blocks_per_stage, dropout,
    ):
        """Build tapering dense network with dimension transitions.

        Creates a network that tapers through decreasing hidden
        dimensions (e.g. [512, 256, 128]), with residual blocks at
        each stage and linear resize layers between stages.
        """
        layers = [
            torch.nn.Linear(in_dim, hidden_dims[0]),
            torch.nn.GELU(),
        ]

        for stage_idx, dim in enumerate(hidden_dims):
            # Residual blocks at this width
            for _ in range(blocks_per_stage):
                layers.append(
                    DenseResidualBlock(dim, dropout)
                )

            # Resize to next stage (if not the last)
            if stage_idx < len(hidden_dims) - 1:
                next_dim = hidden_dims[stage_idx + 1]
                layers.append(torch.nn.Linear(dim, next_dim))
                layers.append(torch.nn.GELU())

        # Final projection to embedding dim
        layers.append(torch.nn.Linear(hidden_dims[-1], out_dim))
        return layers

    @staticmethod
    def _load_svd_bases(path, num_branches):
        """Load precomputed V matrices and n_freq from HDF5."""
        if path is None:
            return None, None
        import numpy as np
        import h5py
        V_matrices = []
        n_freqs = []
        with h5py.File(path, "r") as f:
            for i in range(num_branches):
                key = f"branch_{i}"
                if key in f:
                    V_matrices.append(np.array(f[key]["V"]))
                    n_freqs.append(int(f[key].attrs["n_freq"]))
                else:
                    V_matrices.append(None)
                    n_freqs.append(0)
        return V_matrices, n_freqs

    def set_svd_frozen(self, frozen: bool):
        """Freeze or unfreeze all SVD projection layers."""
        for svd_layer in self.svd_layers:
            if frozen:
                svd_layer.freeze()
            else:
                svd_layer.unfreeze()

    def forward(self, *segments):
        embeddings = []
        for i, seg in enumerate(segments):
            # FFT + SVD projection -> normalize -> dense embedding
            svd_coeffs = self.svd_layers[i](seg)
            svd_coeffs = self.svd_norms[i](svd_coeffs)
            embeddings.append(self.branches[i](svd_coeffs))
        concat = torch.cat(embeddings, dim=-1)
        return self.classifier(concat).squeeze(-1)


class SupervisedTimeSpectrogramResNet(SupervisedArchitecture):
    """
    Spectrogram and Time Domain ResNet that processes a combination of
    timeseries and spectrogram image data.
    """

    def __init__(
        self,
        num_ifos: int,
        time_classes: int,
        spec_classes: int,
        time_layers: list[int],
        spec_layers: list[int],
        time_kernel_size: int = 3,
        spec_kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        time_norm_layer: Optional[NormLayer] = None,
        spec_norm_layer: Optional[NormLayer] = None,
        **kwargs,
    ):
        super().__init__()
        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=time_layers,
            classes=time_classes,
            kernel_size=time_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=time_norm_layer,
        )

        self.spectrogram_resnet = ResNet2D(
            in_channels=num_ifos,
            layers=spec_layers,
            classes=spec_classes,
            kernel_size=spec_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=spec_norm_layer,
        )

    def forward(self, X, X_spec):
        time_domain_output = self.time_domain_resnet(X)
        spec_domain_output = self.spectrogram_resnet(X_spec)
        return time_domain_output, spec_domain_output


class SupervisedHeterodyneTimeDomainResNet(SupervisedArchitecture):
    """
    Time Domain ResNet that processes a Heterodyned timeseries.

    Args:
        num_chirp_masses (int):
            Number of chirp masses used to define the input channel
            dimension (in_channels = num_ifos x num_chirp_masses).
    """

    def __init__(
        self,
        num_ifos: int,
        num_chirp_masses: int,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos * num_chirp_masses,
            layers=layers,
            classes=1,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

    def forward(self, X):
        return self.time_domain_resnet(X)
