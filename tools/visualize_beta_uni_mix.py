import torch
import matplotlib.pyplot as plt

z= 0.1
p_i = [1-z, z]
n = 10000

alpha = 0.5
beta = 0.5

beta_distribution = torch.distributions.beta.Beta(alpha, beta)
uniform_distribution = torch.distributions.uniform.Uniform(0,1)

x = []
for i in range(n):
    prob = torch.rand(1)
    if prob < p_i[0]:
      # Draw from beta
      x.append(beta_distribution.rsample().item())
    else:
      x.append(uniform_distribution.rsample().item())

a = torch.arange(0, 1, 0.001)
lala = beta_distribution.log_prob(a).exp()
lala2 = uniform_distribution.log_prob(a).exp()
y = p_i[0] * beta_distribution.log_prob(a).exp() + p_i[1] * uniform_distribution.log_prob(a).exp()

log_y = y.log()
orig_y = log_y.exp()

fig, ax = plt.subplots(figsize=(8, 4))

# ax.hist(x, bins=100, density=True)
#ax.plot(a, log_y)
ax.plot(a, orig_y)
plt.ylabel('PDF')
plt.xlabel('Action space')
plt.title(f'Beta distribution $\\alpha$:{alpha}, $\\beta$:{beta}')

fig.tight_layout()

# Save the figure
plt.savefig(r'../images/beta_0.png', dpi=300)
plt.show()
print('Test')
