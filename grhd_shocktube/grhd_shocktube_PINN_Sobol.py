import torch
import torch.nn as nn
import numpy as np
from scipy.stats import qmc
import pytorch_optimizer

## Network architecture
class GRHDPINN(nn.Module):
    def __init__(self, hidden_size=32, n_hidden=5):
        super().__init__()
        layers = [nn.Linear(2, hidden_size), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh()])
        self.network = nn.Sequential(*layers)
        self.head_rho = nn.Linear(hidden_size, 1)
        self.head_v = nn.Linear(hidden_size, 1)
        self.head_p = nn.Linear(hidden_size, 1)

    def forward(self, x, t):
        features = self.network(torch.cat([x, t], dim=1))
        rho = nn.functional.softplus(self.head_rho(features)) + 1e-6
        p = nn.functional.softplus(self.head_p(features)) + 1e-6
        v = torch.tanh(self.head_v(features))
        return rho, v, p

## Loss function
def compute_grhd_loss(model, X, X_0, output_ic, gamma, N_t, N_x):

    t = X[:, 0:1]
    x = X[:, 1:2]

    rho, v, p = model(x, t)

    W = 1.0 / torch.sqrt(1 - v**2)

    ## compute conserved variables
    D = rho * W
    S_x = v*(rho+ p*gamma/(gamma-1)) * W**2
    tau = (rho + p*gamma/(gamma-1)) * W**2 - p

    ## compute fluxes
    F_D = D*v
    F_S_x = S_x*v + p
    F_tau = (tau + p)*v

    ## compute derivatives
    dD_dt = grad_y(D, t)
    dS_x_dt = grad_y(S_x, t)
    dtau_dt = grad_y(tau, t)

    dF_D_dx = grad_y(F_D, x)
    dF_S_x_dx = grad_y(F_S_x, x)
    dF_tau_dx = grad_y(F_tau, x)

    d_rho_dx = grad_y(rho, x)
    d_v_dx = grad_y(v, x)
    d_p_dx = grad_y(p, x)

    d_rho_dx_safe = torch.abs(d_rho_dx).clamp_min(1e-8)
    d_v_dx_safe = torch.abs(d_v_dx).clamp_min(1e-8)
    d_p_dx_safe = torch.abs(d_p_dx).clamp_min(1e-8)

    alpha_rho, alpha_v, alpha_p = 1.0,1.0,1.0
    beta_rho, beta_v, beta_p = 1.0,1.0,1.0
    Lambda = 1.0 /(1.0 + (alpha_rho * torch.abs(d_rho_dx_safe)**beta_rho 
                    + alpha_v * torch.abs(d_v_dx_safe)**beta_v 
                    + alpha_p * torch.abs(d_p_dx_safe)**beta_p)).view(N_t,N_x,1)
    
    ## losses 
    L_t_1 = (dD_dt + dF_D_dx).pow(2).view(N_t,N_x,1)
    L_t_2 = (dS_x_dt + dF_S_x_dx).pow(2).view(N_t,N_x,1)
    L_t_3 = (dtau_dt + dF_tau_dx).pow(2).view(N_t,N_x,1)

    ## total loss
    L_t = torch.mean(Lambda * (L_t_1 + L_t_2 + L_t_3), dim=1)

    ## compute loss for initial conditions
    prediction_tmin = (model(X_0[:, 1:2], X_0[:, 0:1]))

    w_rho, w_v, w_p = 1e4, 1e4, 1e4
    w_r = 1.0

    L_ic_rho = w_rho * torch.square(output_ic[:, 0:1] - prediction_tmin[0]).mean()
    L_ic_v = w_v * torch.square(output_ic[:, 1:2] - prediction_tmin[1]).mean()
    L_ic_p = w_p * torch.square(output_ic[:, 2:3] - prediction_tmin[2]).mean()
    L_ic = L_ic_rho + L_ic_v + L_ic_p

    L_t = torch.cat((L_ic.view(-1,1), w_r*L_t[1:]), dim=0)

    ## causality condition
    if epsilon_t != 0.0:
        zeros_t = torch.zeros(1, 1, device=X.device, dtype=L_t.dtype)
        L_t_shifted = torch.cat((zeros_t, L_t[:-1]), dim=0)
        L_t_cumsum = torch.cumsum(L_t_shifted, dim=0)
        w_t = torch.exp(-epsilon_t * L_t_cumsum)
        L_total = (w_t * L_t).mean()
    else:
        L_total = L_t.mean()

    return L_total

## Gradients by autograd
def grad_y(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

## parameters for the sod shock tube problem
gamma = 5.0 / 3.0

epsilon_t = 1.0

tmin, tmax = 0.0, 0.5
xmin, xmax = 0.0, 1.0

N_t = 2**8
N_x = 2**7
N_0 = 2**6

X_list = []

sampler = qmc.Sobol(d=1, scramble=False)
sampler = sampler.random_base2(m=int(np.log2(N_t)))
l_bounds, u_bounds = [tmin], [tmax]
sample_scaled = qmc.scale(sampler, l_bounds, u_bounds)
t = torch.tensor(sample_scaled, dtype=torch.float32)

for value in t:
    sampler = qmc.Sobol(d=1, scramble=False)
    sampler = sampler.random_base2(m=int(np.log2(N_x)))
    l_bounds, u_bounds = [xmin], [xmax]
    sample_scaled = qmc.scale(sampler, l_bounds, u_bounds)
    x = torch.tensor(sample_scaled, dtype=torch.float32)
    t_repeated = torch.tensor(float(value.detach().cpu().numpy())).repeat(x.shape[0], 1 )
    X_list.append(torch.cat([t_repeated, x], dim=1))
X = torch.cat(X_list, dim=0)
X.requires_grad_(True)

## Generate initial data
t_0 = torch.tensor(tmin).repeat((N_0, 1)).view(-1, 1)
sampler = qmc.Sobol(d=1, scramble=False)
sampler = sampler.random_base2(m=int(np.log2(N_0)))
l_bounds, u_bounds = [xmin], [xmax]
sample_scaled = qmc.scale(sampler, l_bounds, u_bounds)
x_0 = torch.tensor(sample_scaled, dtype=torch.float32)
X_0 = torch.cat((t_0, x_0), dim=1)
X_0.requires_grad_(True)

## Initial conditions
rho_L, rho_R = 1.0, 0.125
p_L, p_R = 1.0, 0.1
v_L, v_R = 0.5, 0.5

x_numpy = X_0[:, 1:2].detach().cpu().numpy()

ic_rho = lambda x: rho_L * (x<=0.5) + rho_R * (x>0.5)
ic_p = lambda x: p_L * (x<=0.5) + p_R * (x>0.5)
ic_v = lambda x: v_L * (x<=0.5) + v_R * (x>0.5)

W_tensor = torch.tensor(1/(1-(ic_v(x_numpy)**2))**(1/2), requires_grad=True)

rho_tensor = torch.tensor(ic_rho(x_numpy), dtype=torch.float32, requires_grad=True )
p_tensor = torch.tensor(ic_p(x_numpy), dtype=torch.float32, requires_grad=True )
v_tensor = torch.tensor(ic_v(x_numpy), dtype=torch.float32, requires_grad=True )

output_ic = torch.cat([rho_tensor, v_tensor, p_tensor], dim=1)

## Training with best-model tracking
model = GRHDPINN(hidden_size=50, n_hidden=6)
optimizer = pytorch_optimizer.SOAP(model.parameters(), lr=1e-04)

best_loss = float('inf')
best_model_state = None

loss_history = []

print('Starting SOAP training...')
for epoch in range(300):
    optimizer.zero_grad()
    loss = compute_grhd_loss(model, X, X_0, output_ic, gamma, N_t, N_x)
    loss.backward()
    optimizer.step()
    
    loss_val = loss.item()
    loss_history.append(loss_val)
    if loss_val < best_loss:
        best_loss = loss_val
        best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch:04d} | Loss = {loss_val:.6f} | Best = {best_loss:.6f}')

print(f'\nSOAP training complete. Best loss = {best_loss:.6f}')
print('Restoring best model before L-BFGS...')
model.load_state_dict(best_model_state)

print('Running L-BFGS refinement (up to 50 iterations)...')
optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=50, history_size=20, line_search_fn='strong_wolfe')

def closure():
    optimizer_lbfgs.zero_grad()
    loss = compute_grhd_loss(model, X, X_0, output_ic, gamma, N_t, N_x)
    loss.backward()
    return loss

lbfgs_loss_before = closure().item()
print(f'L-BFGS starting loss = {lbfgs_loss_before:.6f}')

optimizer_lbfgs.step(closure)

lbfgs_loss_after = closure().item()
print(f'L-BFGS final loss = {lbfgs_loss_after:.6f}')

if lbfgs_loss_after < best_loss:
    print('L-BFGS improved the model!')
    best_loss = lbfgs_loss_after
else:
    print(f'L-BFGS degraded the model. Restoring best model.')
    model.load_state_dict(best_model_state)

## Save the best model
torch.save({
    'epoch': epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
}, 'checkpoint_gamma_5_3.pth')

## Save loss history
np.save('loss_history_gamma_5_3.npy', np.array(loss_history))


