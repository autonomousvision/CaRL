import torch
from matplotlib import pyplot as plt


def hl_gaus(mean, std, vmin, vmax, bucket_size):
  bins = torch.arange(vmin-(bucket_size*0.5), vmax, bucket_size)
  bins2 = torch.arange(vmin+(bucket_size*0.5), vmax+(bucket_size*0.5)+0.0001, bucket_size)
  bins3 = torch.arange(vmin, vmax+bucket_size, bucket_size)
  distr = torch.distributions.normal.Normal(mean, std)
  cdf = distr.cdf(bins)
  cdf2 = distr.cdf(bins2)

  pdf = cdf2 - cdf

  return pdf, bins3

mean = torch.tensor(4.0)
std = torch.tensor(0.75)
bins = torch.arange(-10.5, 10.0, 1.0)
bins2 = torch.arange(-9.5, 10.6, 1.0)
bins3 = torch.arange(-10.0, 11.0, 1.0)
distr = torch.distributions.normal.Normal(mean, std)
cdf = distr.cdf(bins)
cdf2 = distr.cdf(bins2)

pdf = cdf2 - cdf
plt.bar(bins3.numpy(), pdf.numpy())
plt.show()

print(torch.sum(pdf))
restored_mean = torch.sum(bins3 * pdf)
print(restored_mean)

pdf2, bins4 = hl_gaus(mean, std, -5.6, 7.8, 0.1)
plt.bar(bins4.numpy(), pdf2.numpy())
plt.show()

restored_mean = torch.sum(pdf2 * bins4)
print(restored_mean)
