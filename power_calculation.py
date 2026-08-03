import numpy as np
from pingouin import power_ttest

# Nigg et al. (2024) meta-analytic estimate
g = 0.249

n = power_ttest(d=g, power=0.80, alpha=0.05, contrast='paired')
print(f"Paired comparison, g = {g}, 80% power, alpha = .05: n = {n:.0f}")

# What could this study have detected?
for n_obs in [18, 20]:
    d = power_ttest(n=n_obs, power=0.80, alpha=0.05, contrast='paired')
    print(f"With n = {n_obs}: minimum detectable d = {d:.3f}")