import numpy as np
import matplotlib.pyplot as plt

# --- パラメータ ---
N = 200
steps = 5000
mu = 0.2

eps_values = np.linspace(0.05,1.0,20)

opinions_list = []

for eps in eps_values:

    opinions = np.random.uniform(-1,1,N)

    for _ in range(steps):

        i,j = np.random.randint(0,N,2)

        d = abs(opinions[i] - opinions[j])

        # interaction
        if d < eps:
            oi = opinions[i]
            oj = opinions[j]

            opinions[i] += mu*(oj-oi)
            opinions[j] += mu*(oi-oj)


    opinions_list.append(opinions)


# --- グラフ ---
eps_values4plot = np.tile(eps_values,len(opinions_list[0]))
eps_values4plot = eps_values4plot.flatten()
opinions4plot = np.array(opinions_list).T.flatten()

plt.figure()
plt.plot(eps_values4plot,opinions4plot,'.')
plt.xlabel("filter threshold ε")
plt.ylabel("opinion")
plt.title('SNS opinions')
plt.show()
