import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels = 1024):
        super().__init__()
        modules = []

        modules.append(nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=2, stride=1, padding=1))
        modules.append(torch.nn.BatchNorm1d(64))
        modules.append(torch.nn.LeakyReLU(0.2, inplace=True))
        modules.append(nn.Conv1d(64, 128, kernel_size=2, stride=1, padding=1))
        modules.append(torch.nn.BatchNorm1d(128))
        modules.append(torch.nn.LeakyReLU(0.2, inplace=True))
        modules.append(nn.Conv1d(128, 256, kernel_size=2, stride=1, padding=1))
        modules.append(torch.nn.BatchNorm1d(256))
        modules.append(torch.nn.LeakyReLU(0.2, inplace=True))
        modules.append(nn.Conv1d(256, 512, kernel_size=2, stride=1, padding=1))
        modules.append(torch.nn.BatchNorm1d(512))
        modules.append(torch.nn.LeakyReLU(0.2, inplace=True))
        modules.append(nn.Conv1d(512, out_channels, kernel_size=2, stride=1, padding=1))
        modules.append(torch.nn.BatchNorm1d(out_channels)) #1024 x 314( stride 1 minus 2 and approx 307-310)
        self.cnn = nn.Sequential(*modules)#.double() #נbarak change (double)

    def forward(self, x):
        return self.cnn(x)#.double()


class DecoderCNN(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1):
        super().__init__()
        modules = []
        modules.append(torch.nn.ReLU(inplace=True))
        modules.append(torch.nn.BatchNorm1d(in_channels))
        modules.append(nn.ConvTranspose1d(in_channels, 512, kernel_size=2, stride=1, padding=1))
        modules.append(torch.nn.ReLU(inplace=True))
        modules.append(torch.nn.BatchNorm1d(512))
        modules.append(nn.ConvTranspose1d(512, 256, kernel_size=2, stride=1, padding=1))
        modules.append(torch.nn.ReLU(inplace=True))
        modules.append(torch.nn.BatchNorm1d(256))
        modules.append(nn.ConvTranspose1d(256, 128, kernel_size=2, stride=1, padding=1))
        modules.append(torch.nn.ReLU(inplace=True))
        modules.append(torch.nn.BatchNorm1d(128))
        modules.append(nn.ConvTranspose1d(128, 64, kernel_size=2, stride=1, padding=1))
        modules.append(torch.nn.ReLU(inplace=True))
        modules.append(torch.nn.BatchNorm1d(64))
        modules.append(nn.ConvTranspose1d(64, out_channels, kernel_size=2, stride=1, padding=1, output_padding=0))
        self.cnn = nn.Sequential(*modules)#.double()

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))#.double()


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from it's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim
        self.features_shape, n_features = self._check_features(in_size)
        self.fc_mean = torch.nn.Linear(n_features, self.z_dim)
        self.fc_log_variance = torch.nn.Linear(n_features, self.z_dim)
        self.fc_z_dim_backTo_n_features = torch.nn.Linear(self.z_dim, n_features)


    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # we make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device) #,dtype=torch.double
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]


    def encode(self, x):
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  Use the features extracted from the input to obtain mu and log_sigma2 (mean and log variance) of q(Z|x).
        #  Apply the reparametrization trick to obtain z.
        h = self.features_encoder(x)
        h = h.reshape((h.shape[0], -1))
        mu = self.fc_mean(h)
        log_sigma2 = self.fc_log_variance(h)
        u = torch.normal(torch.zeros_like(mu))
        z = mu + u.mul(log_sigma2.exp())
        #z = z.double() #barak change
        return z, mu, log_sigma2

    def decode(self, z):
        # Convert a latent vector back into a reconstructed input.
        # Convert latent z to features h with a linear layer.
        # Apply features decoder.
        h = self.fc_z_dim_backTo_n_features(z)
        h_r = h.view(-1, *self.features_shape)
        x_rec = self.features_decoder(h_r)
        return torch.tanh(x_rec)#.double()# Scale to [-1, 1] (same dynamic range as original images).

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            #  Sample from the model.
            #  Generate n latent space samples and return their reconstructions.
            #  Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #  the mean, i.e. psi(z).
            latent_space_samples = torch.randn(n, self.z_dim).to(device)
            samples = self.decode(latent_space_samples).cpu()
        return samples

    def forward(self, x):#4,314 batch, first dim of data,, 314
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    z_sigma2 = torch.exp(z_log_sigma2)
    data_loss = F.mse_loss(x, xr).mul(1 / x_sigma2)
    kldiv_loss = (z_sigma2 + z_mu.pow(2) - 1 - z_log_sigma2).sum(dim=1)
    kldiv_loss = torch.mean(kldiv_loss)
    loss = data_loss + kldiv_loss
    return loss, data_loss, kldiv_loss

