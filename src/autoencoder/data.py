"""Fashion-MNIST: leitura IDX, separação one-class e lotes PyTorch.

Os índices de treino/validação referem-se aos 60 mil exemplos oficiais.
Os índices de teste referem-se a OUTRO arquivo, com 10 mil exemplos.
Rótulos servem para filtrar/avaliar; o alvo de reconstrução é a própria imagem.
"""

from dataclasses import dataclass
import gzip
import hashlib
from numbers import Integral
from pathlib import Path
import random
import shutil
import struct
import tempfile
from urllib.request import urlopen

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
# Checksums publicados no README oficial da Zalando (integridade dos dados).
MD5 = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


@dataclass(frozen=True)
class DataConfig:
    directory: Path
    anomaly_class: int = 8
    validation_fraction: float = 0.15
    seed: int = 42
    batch_size: int = 128
    num_workers: int = 0

    def __post_init__(self):
        _validate_split_options(self.anomaly_class, self.validation_fraction, self.seed)
        _validate_loader_options(self.batch_size, self.num_workers)


@dataclass(frozen=True)
class DataSplit:
    images: np.ndarray  # float32, (N, 28, 28), intervalo [0, 1]
    labels: np.ndarray  # int64, classes originais 0..9
    indices: np.ndarray  # posição no arquivo indicado por source
    source: str  # official_train ou official_test


@dataclass(frozen=True)
class FashionSplits:
    train: DataSplit
    val: DataSplit
    test: DataSplit
    anomaly_class: int
    seed: int
    validation_fraction: float


def _validate_split_options(anomaly_class, validation_fraction, seed):
    if isinstance(anomaly_class, bool) or not isinstance(anomaly_class, Integral) or not 0 <= anomaly_class < 10:
        raise ValueError("anomaly_class deve ser um inteiro de 0 a 9")
    if isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction deve estar entre 0 e 1")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed deve ser um inteiro entre 0 e 2**32 - 1")


def _validate_loader_options(batch_size, num_workers):
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError("batch_size deve ser um inteiro positivo")
    if isinstance(num_workers, bool) or not isinstance(num_workers, Integral) or num_workers < 0:
        raise ValueError("num_workers deve ser um inteiro não negativo")


def load_config(path: str | Path) -> DataConfig:
    """Lê configs/default.yaml; data.directory é relativa à raiz do projeto.

    A raiz é a pasta pai de configs/, independentemente do diretório atual.
    Caminhos absolutos também são aceitos.
    """
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("A configuração deve conter um mapeamento 'data'")
    data = dict(config["data"])
    directory = Path(data.pop("directory", "data"))
    if not directory.is_absolute():
        directory = path.parent.parent / directory
    return DataConfig(directory=directory, seed=config.get("seed", 42), **data)


def _check_file(path: Path, expected_md5: str) -> None:
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "md5").hexdigest()
    if actual != expected_md5:
        raise ValueError(f"Checksum inválido em {path.name}; remova o arquivo e execute novamente")


def download_data(data_dir: str | Path = "data") -> Path:
    """Baixa somente arquivos ausentes, verifica MD5 e instala atomicamente.

    Um arquivo existente corrompido gera erro explícito, sem ser sobrescrito.
    Downloads interrompidos nunca são reutilizados como arquivos completos.
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    for filename in FILES.values():
        destination = root / filename
        if destination.exists():
            _check_file(destination, MD5[filename])
            continue
        with tempfile.NamedTemporaryFile(dir=root, suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with urlopen(BASE_URL + filename, timeout=60) as response, temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            _check_file(temporary, MD5[filename])
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return root


def read_idx(path: str | Path) -> np.ndarray:
    """Lê os IDX gzip de bytes sem sinal usados pelo Fashion-MNIST.

    Verifica cabeçalho, dimensões e tamanho exato; rejeita arquivos truncados.
    """
    with gzip.open(path, "rb") as handle:
        header = handle.read(4)
        if len(header) != 4 or header[:3] != b"\x00\x00\x08" or header[3] not in (1, 3):
            raise ValueError("Cabeçalho IDX inválido: esperado uint8 com 1 ou 3 dimensões")
        ndim = header[3]
        dimensions = handle.read(4 * ndim)
        if len(dimensions) != 4 * ndim:
            raise ValueError("Dimensões IDX truncadas")
        shape = struct.unpack(">" + "I" * ndim, dimensions)
        if any(size == 0 for size in shape):
            raise ValueError("Dimensões IDX devem ser positivas")
        payload = handle.read()
    if len(payload) != int(np.prod(shape, dtype=object)):
        raise ValueError("Tamanho IDX incompatível com o cabeçalho")
    return np.frombuffer(payload, dtype=np.uint8).reshape(shape).copy()


def _validate_raw(raw: dict[str, np.ndarray]) -> None:
    for partition in ("train", "test"):
        images, labels = raw[f"{partition}_images"], raw[f"{partition}_labels"]
        if images.ndim != 3 or images.shape[1:] != (28, 28) or images.dtype != np.uint8:
            raise ValueError("Imagens devem ter formato (N, 28, 28) e dtype uint8")
        if labels.ndim != 1 or len(images) != len(labels) or not len(labels):
            raise ValueError("Quantidade/formato de imagens e rótulos incompatível")
        if not np.issubdtype(labels.dtype, np.integer) or np.any((labels < 0) | (labels > 9)):
            raise ValueError("Rótulos devem ser inteiros de 0 a 9")


def load_raw(data_dir: str | Path = "data", download: bool = True) -> dict[str, np.ndarray]:
    """Carrega os quatro arquivos oficiais, verificando integridade e formato."""
    root = download_data(data_dir) if download else Path(data_dir)
    if not download:
        for filename in FILES.values():
            _check_file(root / filename, MD5[filename])
    raw = {key: read_idx(root / filename) for key, filename in FILES.items()}
    _validate_raw(raw)
    if len(raw["train_labels"]) != 60_000 or len(raw["test_labels"]) != 10_000:
        raise ValueError("Esperados 60.000 exemplos oficiais de treino e 10.000 de teste")
    return raw


def make_splits(
    raw: dict[str, np.ndarray],
    anomaly_class: int = 8,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> FashionSplits:
    """Filtra anomalias ANTES de dividir o treino oficial por classe.

    Os 54.000 normais tornam-se 45.900 de treino e 8.100 de validação.
    Os 6.000 anômalos do treino oficial ficam excluídos de ambos.
    O teste preserva a ordem e TODAS as classes dos 10.000 exemplos oficiais.
    """
    _validate_split_options(anomaly_class, validation_fraction, seed)
    _validate_raw(raw)
    labels = raw["train_labels"]
    eligible = np.flatnonzero(labels != anomaly_class)
    train_indices, val_indices = train_test_split(
        eligible, test_size=validation_fraction, random_state=int(seed),
        stratify=labels[eligible],
    )

    def prepare(partition, indices, source):
        return DataSplit(
            images=raw[f"{partition}_images"][indices].astype(np.float32) / np.float32(255),
            labels=raw[f"{partition}_labels"][indices].astype(np.int64),
            indices=indices.copy(), source=source,
        )

    return FashionSplits(
        train=prepare("train", train_indices, "official_train"),
        val=prepare("train", val_indices, "official_train"),
        test=prepare("test", np.arange(len(raw["test_labels"])), "official_test"),
        anomaly_class=int(anomaly_class), seed=int(seed),
        validation_fraction=validation_fraction,
    )


def load_splits(config: DataConfig, download: bool = True) -> FashionSplits:
    """Entrada principal para os scripts de treino/avaliação."""
    return make_splits(
        load_raw(config.directory, download=download), config.anomaly_class,
        config.validation_fraction, config.seed,
    )


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloaders(
    splits: FashionSplits, batch_size: int = 128, num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Produz (imagem, classe_original); imagem tem forma (N, 1, 28, 28).

    Somente treino embaralha. Geradores locais tornam a sequência reproduzível
    entre loaders novos com a mesma seed; novas épocas avançam esse gerador.
    Para o Autoencoder, compare a reconstrução com a imagem, não com a classe.
    """
    _validate_loader_options(batch_size, num_workers)
    loaders = {}
    for name in ("train", "val", "test"):
        split = getattr(splits, name)
        dataset = TensorDataset(
            torch.from_numpy(split.images[:, None, :, :]), torch.from_numpy(split.labels),
        )
        loaders[name] = DataLoader(
            dataset, batch_size=batch_size, shuffle=name == "train", drop_last=False,
            num_workers=num_workers, worker_init_fn=_seed_worker,
            generator=torch.Generator().manual_seed(splits.seed),
        )
    return loaders
