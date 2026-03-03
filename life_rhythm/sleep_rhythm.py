import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib_fontja

np.random.seed(0)

days = 30
hours = 24

def generate_life_pattern(type_name):
    data = []
    for d in range(days):
        if type_name == "regular":
            sleep_start = 23
            sleep_end = 7
        elif type_name == "slightly_irregular":
            sleep_start = (23 + np.random.choice([-1, 0, 1])) % 24
            sleep_end = (7 + np.random.choice([-1, 0, 1])) % 24
        else:  # highly_irregular
            sleep_start = np.random.randint(0, 24)
            sleep_length = np.random.randint(5, 9)
            sleep_end = (sleep_start + sleep_length) % 24

        row = []
        for h in range(hours):
            if sleep_start < sleep_end:
                asleep = sleep_start <= h < sleep_end
            else:
                asleep = h >= sleep_start or h < sleep_end
            row.append(0 if asleep else 1)
        data.append(row)
    return pd.DataFrame(data, columns=[f"h{h}" for h in range(hours)])

regular = generate_life_pattern("regular")
slightly = generate_life_pattern("slightly_irregular")
irregular = generate_life_pattern("highly_irregular")

print(regular.head())
print(slightly.head())
print(irregular.head())

print(np.array(regular).flatten())
print(np.array(slightly).flatten())
print(np.array(irregular).flatten())

plt.figure(figsize=(10,4))
plt.suptitle('生活リズム')
plt.axis('off')
plt.subplot(1,3,1)
plt.title('規則的')
sns.heatmap(regular, cbar=False)
plt.xlabel('時間')
plt.ylabel('日にち')
plt.subplot(1,3,2)
plt.title('正常なリズム')
sns.heatmap(slightly, cbar=False)
plt.xlabel('時間')
plt.ylabel('日にち')
plt.subplot(1,3,3)
plt.title('異常なリズム')
sns.heatmap(irregular, cbar=False)
plt.xlabel('時間')
plt.ylabel('日にち')
plt.tight_layout()



plt.figure(figsize=(10,4))
plt.suptitle('生活リズム')
plt.axis('off')
plt.subplot(1,3,1)
plt.title('規則的')
r_mean = np.mean(regular,axis=0)
r_std = np.std(regular,axis=0)
plt.errorbar(range(24),r_mean, r_std,capsize=5, fmt='-', markersize=3, ecolor='black', markeredgecolor = "black", color='k')
plt.xlabel('時間')
plt.ylim([-0.2,1.4])
plt.subplot(1,3,2)
plt.title('正常なリズム')
s_mean = np.mean(slightly,axis=0)
s_std = np.std(slightly,axis=0)
plt.errorbar(range(24),s_mean, s_std,capsize=5, fmt='-', markersize=3, ecolor='black', markeredgecolor = "black", color='k')
plt.xlabel('時間')
plt.ylim([-0.2,1.4])
plt.subplot(1,3,3)
plt.title('異常なリズム')
i_mean = np.mean(irregular,axis=0)
i_std = np.std(irregular,axis=0)
plt.errorbar(range(24),i_mean, i_std,capsize=5, fmt='-', markersize=3, ecolor='black', markeredgecolor = "black", color='k')
plt.xlabel('時間')
plt.ylim([-0.2,1.4])
plt.tight_layout()
plt.show()

print(f"規則的:{np.mean(r_std)}")
print(f"正常　:{np.mean(s_std)}")
print(f"異常　:{np.mean(i_std)}")