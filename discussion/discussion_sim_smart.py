import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# パラメータ
# -----------------------------
N = 10 # エージェント数
T = 100 # 議論ステップ数

alpha = 0.4 # 理解学習率
beta = 0.3 # 意見更新率
gamma = 5.0 # 理解度の進み方（意見が予想とズレた時、小さいと理解度が下がりずらく、大きいと理解度が下がりやすい）
sigma = 0.1 # 表現ノイズ
agree_ratio = 0.4 # 賛成率
disagree_ratio = 0.6 # 反対率 
understand_ratio = 1 - agree_ratio - disagree_ratio

np.random.seed(0)

# -----------------------------
# 初期化
# -----------------------------
x = np.random.rand(N) # 意見
u = np.zeros((N, N))  # 理解度
np.fill_diagonal(u, 1.0) # 自己理解=1

x_hat = np.random.rand(N,N) # i->jの相手意見の予測

x_history = np.zeros((T, N))
U_history = []

# -----------------------------
# シミュレーション
# -----------------------------

reactivity = np.random.choice([1.0, 0.0, -1.0], size=N, p=[agree_ratio, understand_ratio,disagree_ratio])
for t in range(T):
    for _ in range(N):  # ランダムなペアで議論
        i, j = np.random.choice(N, 2, replace=False)

        # i が発言
        s = x[i] + np.random.normal(0, sigma)

        # 相手モデル更新（理解）
        x_hat[j, i] += alpha * (s - x_hat[j, i])

        # j の予測（単純化：現在の意見）
        pred = x_hat[j,i]

        # 予測誤差
        error = abs(s - pred)

        # --- 理解更新 ---
        # 理解度
        u[j, i] = np.exp(-gamma * (s - pred)**2)

        # 意見更新（反応タイプ）
        x[j] += beta * u[j, i] * reactivity[j] * (s - x[j])
        x = np.clip(x,0,1)

    # 指標記録
    x_history[t] = x
    U_history.append(np.mean(u[np.eye(N) == 0]))

# -----------------------------
# 可視化
# -----------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(U_history)
ax[0].set_title("Average Mutual Understanding")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("U")
ax[0].set_ylim(0, 1.1)


for i in range(N):
    ax[1].plot(x_history[:, i], lw=2)

ax[1].set_xlabel("Time")
ax[1].set_ylabel("Opinion x")
ax[1].set_title("Opinion Trajectories (Understanding-based Update)")
ax[1].set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()