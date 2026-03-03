import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# パラメータ
# -----------------------------
N = 10 # エージェント数
T = 200 # 議論ステップ数

alpha = 0.5 # 理解学習率
beta = 0.3 # 意見更新率
gamma = 5.0 # 理解度の進み方（意見が予想とズレた時、小さいと理解度が下がりずらく、大きいと理解度が下がりやすい）
sigma = 0.1 # 表現ノイズ
agree_ratio = 0.4 # 賛成率
disagree_ratio = 0.6 # 反対率 
understand_ratio = 1 - agree_ratio - disagree_ratio # 理解はするが意見は変えない率
k = 3 # ストレスモデルの係数
recovery = 0.008 # ストレスの回復係数
c = 1 # 意見距離で増えるストレスの係数
d = 1 # 理解度で減るストレスの係数
eta = 1 # ストレスでの発言の誇張度

np.random.seed(5)

# -----------------------------
# 初期化
# -----------------------------
x = np.random.rand(N) # 意見
u = np.zeros((N, N))  # 理解度
np.fill_diagonal(u, 1.0) # 自己理解=1

x_hat = np.random.rand(N,N) # i->jの相手意見の予測
z = np.random.gamma(shape=2.0, scale=0.5, size=(N, N)) # j が i と話すときのストレス
np.fill_diagonal(z, 0) # 自己ストレス=0
stubborn = np.ones(N) # 意見の聞きやすさ

# 意固地な人の設定
#stubborn_num = None
#stubborn_num = [N-1]
stubborn_num = [N-1,N-2]
#stubborn_num = [N-1,N-2,N-3]
#stubborn_num = [N-1,N-2,N-3,N-4]

if stubborn_num:
    x[stubborn_num] = np.random.choice([0.0,1.0],size=len(stubborn_num))# 意固地な人
    stubborn[stubborn_num] = 0.001  # 意固地な人


x_history = np.zeros((T, N))
U_history = []
z_history = []
z_stubborn_history = []

reactivity = np.random.choice([1.0, 0.0, -1.0], size=N, p=[agree_ratio, understand_ratio,disagree_ratio])

# -----------------------------
# シミュレーション
# -----------------------------

for t in range(T):
    for _ in range(N):  # ランダムなペアで議論
        i, j = np.random.choice(N, 2, replace=False)
        # i が発言
        # ストレスで発言が誇張される
        s = x[i] + np.random.normal(0, sigma) + eta * z[i, j] * (x[i] - 0.5)

        # 相手モデル更新（理解）（ストレスモデル込み）
        alpha_eff = alpha * np.exp(-k * z[j, i]) * stubborn[j]
        x_hat[j, i] += alpha_eff * (s - x_hat[j, i])

        # j の予測（単純化：現在の意見）
        pred = x_hat[j,i]

        # 予測誤差
        error = abs(s - pred)

        # --- 理解更新 ---
        # 理解度
        u[j, i] = np.exp(-gamma * (s - pred)**2)
        # ストレス
        # 意見距離でストレスが増える
        z[j, i] += c * abs(s - x[j])
        # 理解度でストレスが減る
        z[j, i] -= d * u[j, i]
        z[j, i] = max(z[j, i], 0)
        # ストレスの時間減衰
        z *= (1 - recovery)
        # 意見更新（反応タイプ）
        x[j] += beta  * stubborn[j] * u[j, i] * reactivity[j] * (s - x[j])
        x = np.clip(x,0,1)

    # 指標記録
    x_history[t] = x
    U_history.append(np.mean(u[np.eye(N) == 0]))
    z_history.append(np.mean(z[np.eye(N) == 0]))
    if stubborn_num:
        z_stubborn = np.mean(z[stubborn_num, :]) + np.mean(z[:, stubborn_num])
    else:
        z_stubborn = np.mean(z[0, :]) + np.mean(z[:, 0])
    z_stubborn_history.append(z_stubborn)

# -----------------------------
# 可視化
# -----------------------------
fig, ax = plt.subplots(1, 3, figsize=(10, 4))

ax[0].plot(U_history)
ax[0].set_title("Average Mutual Understanding")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("U")
ax[0].set_ylim(0, 1.1)


ax[1].plot(z_history,label='mean')
if stubborn_num:
    ax[1].plot(z_stubborn_history,label='related stubborn')
else:
    ax[1].plot(z_stubborn_history,label='related num 0')
ax[1].legend()
ax[1].set_title("Average Mutual Stress")
ax[1].set_xlabel("Time")
ax[1].set_ylabel("z")
#ax[1].set_ylim(0, 1.1)


for i in range(N):
    if stubborn_num:
        if i in stubborn_num:
            ax[2].plot(x_history[:,i],'--',lw=2)
        else:
            ax[2].plot(x_history[:, i], lw=2)
    else:
        ax[2].plot(x_history[:, i], lw=2)

ax[2].set_xlabel("Time")
ax[2].set_ylabel("Opinion x")
ax[2].set_title("Opinion Trajectories")
ax[2].set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()