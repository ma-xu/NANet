import time
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

mu = torch.FloatTensor([2, 4])
sigma = torch.FloatTensor([[5, 0], [0, 2]])

mu_gpu = mu.cuda()
sigma_gpu = sigma.cuda()

num_runs = 1000
t_cpu, t_gpu = 0, 0
for _ in range(num_runs):
    st = time.perf_counter()
    m1 = MultivariateNormal(mu, sigma)
    t_cpu += time.perf_counter() - st

    torch.cuda.synchronize()
    st = time.perf_counter()
    m2 = MultivariateNormal(mu_gpu, sigma_gpu)
    torch.cuda.synchronize()
    t_gpu += time.perf_counter() - st

print(f'[CPU] Time Taken: {t_cpu}s')
print(f'[GPU] Time Taken: {t_gpu}s')
