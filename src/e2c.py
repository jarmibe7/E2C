"""
e2c.py

Embed to Control model architecture made with PyTorch.

Authors: Jared Berry, Ayush Gaggar
"""
import torch
from torch import nn
from pathlib import Path

from src.encode import ConvEncoder, ConvDecoder

# Get paths relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
CONFIG_PATH = PROJECT_ROOT / "config"

class E2CDataset(torch.utils.data.Dataset):
    """
    An E2C Dataset consists of a current image, a future image, and a control input.
    """
    def __init__(self, config):
        # Load raw dataset
        dataset_dir = DATA_PATH / f"{config['train']['dataset']}"
        img = torch.load(dataset_dir / 'img.pt')
        control = torch.load(dataset_dir / 'img.pt')

        # Reshape for learning
        X = img[:, :-1]   # Shape: [batch, seq_len - 1, H, W, C]
        X_next = img[:, 1:]
        X = X.reshape(-1, X.shape[2], X.shape[3], X.shape[4]).permute(0, 3, 1, 2)   # Shape: [batch, C, H, W]
        X_next = X_next.reshape(-1, X_next.shape[2], X_next.shape[3], X_next.shape[4]).permute(0, 3, 1, 2)
        U = control.reshape(-1, control.shape[-1])
        self.X = X
        self.X_next = X_next
        self.U = U

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.X_next[idx], self.U[idx]
  
class E2CLoss(nn.Module):
    """
    E2C loss, made with PyTorch.
    """
    def __init__(self, num_epochs, loss_params):
        super().__init__()
        self.num_epochs = num_epochs
        self.recon_mult = loss_params['recon_mult']
        self.beta = loss_params['beta']
        self.lam = loss_params['lambda']
        self.anneal_mode = loss_params['kld_anneal_mode']

    def kld_anneal(self, epoch):
        if self.anneal_mode == 'const':
            mult = self.beta
        elif self.anneal_mode == 'linear':
            mult = self.beta*((epoch + 1)/self.num_epochs)
        else:
            raise NotImplementedError(f"Annealing mode {self.anneal_mode} not supported!")

        return mult

    def forward(self, tr, epoch):
        # Reconstruction loss
        recon = self.recon_mult*nn.functional.mse_loss(tr['x'], tr['x_recon'], reduction='mean')
        recon_next = self.recon_mult*nn.functional.mse_loss(tr['x_next'], tr['x_next_recon'], reduction='mean')

        # Encoding KL Divergence
        log_var, mu = tr['log_var'], tr['mu']
        kld = self.kld_anneal(epoch)*(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean())

        # Contraction term on transition posterior
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        log_var_pred, mu_pred = tr['log_var_pred'], tr['mu_pred']
        log_var_next, mu_next = tr['log_var_next'], tr['mu_next']
        var_next, var_pred = torch.exp(log_var_next), torch.exp(log_var_pred)

        # Times 0.5 because formula uses standard dev
        # kld_contract_vec = 0.5*(log_var_pred - log_var_next) \
        #                    + ((var_next + (mu_next - mu_pred)**2) / (var_pred + 1e-8)) \
        #                    - 1.0
        # kld_contract = 0.5 * torch.sum(kld_contract_vec)

        # v, r = tr['v'], tr['r']
        # dot = torch.sum(v * r, dim=1)
        # dot = torch.clamp(dot, min=-0.99)  # Ensure log(1 + dot) > 0
        # sum_term = torch.sum(log_var_pred - log_var_next, axis=1)
        # log_term = torch.log(1 + dot)
        # kld_contract_vec = 2*(sum_term - log_term)
        # kld_contract = self.lam*torch.sum(kld_contract_vec)
        z_pred = tr['z_pred']
        kld_contract = self.lam*(-0.5 * torch.sum(1 + log_var_next - (z_pred - mu_next).pow(2) - log_var_next.exp(), dim=-1).mean())

        loss = recon + recon_next + kld + kld_contract
        if torch.isnan(loss):
            breakpoint()
        return loss, recon, recon_next, kld, kld_contract

class E2C(nn.Module):
    """
    An E2C model with convolutional encoder-decoder.
    """
    def __init__(self, enc_latent_size, latent_size, control_size, conv_params, device):
        super().__init__()
        self.device = device

        # Set number of hidden units
        self.enc_latent_size = enc_latent_size
        self.latent_size = latent_size
        self.control_size = control_size

        # Dummy zero control vector
        self.dummy_u = torch.zeros((1, self.control_size)).to(self.device)

        # Encoder and decoder
        self.encoder = ConvEncoder(enc_latent_size, conv_params)
        self.decoder = ConvDecoder(latent_size, conv_params, self.encoder.out_dim_flat, self.encoder.out_shape)

        # VAE
        self.mu = nn.Linear(self.enc_latent_size, self.latent_size)
        self.log_var = nn.Linear(self.enc_latent_size, self.latent_size)

        # Locally linear transition model
        self.fc_trans = nn.Sequential(
            nn.Linear(self.latent_size, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, 32)
        )
        self.v = nn.Linear(32, self.latent_size)  # A jacobian
        self.r = nn.Linear(32, self.latent_size)
        self.B = nn.Linear(32, self.latent_size*self.control_size)
        self.o = nn.Sequential(
            nn.Linear(32, self.latent_size),
            nn.Tanh()
        )
        self.H = nn.Sequential(
            nn.Linear(32, self.latent_size),
            nn.Softplus() # Ensure positive covariances
        )

    def reparameterize(self, mu, log_var):
        # Get standard deviation from log variance
        std = torch.exp(0.5 * log_var)
        std = torch.clamp(std, min=1e-5, max=1e5) # Prevent std from being too small

        # Generate random noise epsilon of same shape std
        eps = torch.randn_like(std)

        # Return reparameterized sample
        return mu + eps * std

    def transition(self, z, mu, log_var, u):
        # Pass through transition network
        trn = self.fc_trans(z)

        # Get Jacobians and linear model parameters
        v = self.v(trn).unsqueeze(-1)                                                    #   [batch, z, 1]
        r = self.r(trn).unsqueeze(-1)                                                    #   [batch, z, 1]
        A = torch.eye(self.latent_size, device=self.device).repeat(z.size(0), 1, 1) \
            + torch.bmm(v, r.transpose(1, 2))                                               #   [batch, z, z]
        B = self.B(trn).reshape((-1, self.latent_size, self.control_size))               #   [batch, z, u]
        o = self.o(trn).unsqueeze(-1)                                                    #   [batch, z, 1]
        H = torch.diag_embed(self.H(trn))                                                   #   [batch, z, z]

        # Use linear model and reparam to create posterior distribution
        mu_pred = torch.bmm(A, mu.unsqueeze(-1)) + torch.bmm(B, u.unsqueeze(-1)) + o         #   [batch, z, 1]
        mu_pred = mu_pred.squeeze(-1)                                                  #   [batch, z]
        sigma = torch.diag_embed(torch.exp(log_var))                                         #   [batch, z, z]
        C = torch.bmm(A, torch.bmm(sigma, A.transpose(1, 2))) + H                            #   [batch, z, z]
        log_var_pred = torch.log(torch.diagonal(C, dim1=-2, dim2=-1) + 1e-8)                 #   [batch, z]
        z_pred = self.reparameterize(mu_pred, log_var_pred)                            #   [batch, z]

        return mu_pred, log_var_pred, z_pred, v.squeeze(-1), r.squeeze(-1)

    def forward(self, x, x_next, u):
        # Encode current and next state
        enc_out = self.encoder(x)
        enc_out_next = self.encoder(x_next)

        # Get latent variable
        mu = self.mu(enc_out)
        log_var = self.log_var(enc_out)
        z = self.reparameterize(mu, log_var)

        # Get next latent variable
        mu_next = self.mu(enc_out_next)
        log_var_next = self.log_var(enc_out_next)
        z_next = self.reparameterize(mu_next, log_var_next)

        # Transition model
        mu_pred, log_var_pred, z_pred, v, r = self.transition(z, mu, log_var, u)

        # Get reconstruction and prediction
        x_recon = self.decoder(z)
        x_next_recon = self.decoder(z_next)
        x_pred = self.decoder(z_pred)

        train_return = {
            'x_recon': x_recon,
            'mu': mu,
            'log_var': log_var,
            'x_next_recon': x_next_recon,
            'mu_next': mu_next,
            'log_var_next': log_var_next,
            'x_pred': x_pred,
            'z_pred': z_pred,
            'mu_pred': mu_pred,
            'log_var_pred': log_var_pred,
            'v': v,
            'r': r
        }

        return train_return

    def reconstruct(self, x_traj):
        with torch.no_grad():
            frames = []
            for x in x_traj:
                # Encode current state
                encoded = self.encoder(x.unsqueeze(0))
                flattened = encoded.view(encoded.size(0), -1)

                # Get latent variable
                mu = self.mu(flattened)
                log_var = self.log_var(flattened)
                z = self.reparameterize(mu, log_var)
                frames.append(self.decoder(z))

            return torch.concat(frames, dim=0).squeeze(0).to('cpu').permute(0, 2, 3, 1)

    def sample(self, x0, seq_len):
        with torch.no_grad():
            # Encode current state
            encoded = self.encoder(x0.unsqueeze(0).to(self.device))
            flattened = encoded.view(encoded.size(0), -1)

            # Get latent variable
            mu = self.mu(flattened)
            log_var = self.log_var(flattened)
            z = self.reparameterize(mu, log_var)

            # Transition model
            frames = []
            frames.append(self.decoder(z))
            for t in range(seq_len):
                mu, log_var, z, _, _ = self.transition(z, mu, log_var, self.dummy_u)
                frames.append(self.decoder(z))

            return torch.concat(frames, dim=0).squeeze(0).to('cpu').permute(0, 2, 3, 1)
