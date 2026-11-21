import pytest
import torch

from autoencoder import Autoencoder, reconstruction_loss


def test_autoencoder_shapes_and_bounded_output():
    model = Autoencoder(latent_dim=8)
    inputs = torch.rand(4, 784)

    latent = model.encode(inputs)
    reconstruction = model(inputs)

    assert latent.shape == (4, 8)
    assert reconstruction.shape == (4, 784)
    assert torch.all((reconstruction >= 0) & (reconstruction <= 1))


def test_forward_preserves_flat_or_image_input_shape():
    model = Autoencoder(latent_dim=8)

    for inputs in (torch.rand(3, 784), torch.rand(3, 1, 28, 28)):
        assert model(inputs).shape == inputs.shape
        assert model.encode(inputs).shape == (3, 8)


def test_model_reads_latent_dimension_from_yaml():
    model = Autoencoder.from_config("configs/default.yaml")

    assert model.latent_dim == 16
    assert model.encode(torch.rand(2, 784)).shape == (2, 16)


def test_parameter_count_and_mse_loss():
    model = Autoencoder(latent_dim=8)

    assert model.count_parameters() == 436_632
    assert reconstruction_loss(torch.zeros(1, 784), torch.ones(1, 784)).item() == 1


@pytest.mark.parametrize("latent_dim", [0, -1, True, 1.5])
def test_invalid_latent_dimension_is_rejected(latent_dim):
    with pytest.raises(ValueError, match="latent_dim"):
        Autoencoder(latent_dim=latent_dim)
