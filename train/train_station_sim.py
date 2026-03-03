import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# パラメータ
# -----------------------
T = 120                  # シミュレーション時間（分）
N0 = 1200                # 初期混雑人数
N_safe = 200             # 安全水準

lambda_in =  5           # 途中流入（人/分）
mu = 50                  # 乗車能力（人/分）
C = 800                  # 列車定員
headway = 5              # 列車間隔（分）

base_dwell = 1           # 通常停車時間（分）
extra_wait = 2           # 混雑時に追加で待たせる時間（分）
N_th = 400               # 制御発動しきい値

control = False          # 意図的遅延あり / なし

# -----------------------
# 状態変数
# -----------------------
N = N0
N_history = []
delay_sum = 0
clear_time = None

# -----------------------
# シミュレーション
# -----------------------
for t in range(T):
    # 途中流入
    N += lambda_in

    # 列車到着
    if t % headway == 0:
        dwell = base_dwell

        if control and N > N_th:
            dwell += extra_wait
            delay_sum += extra_wait

        board = min(mu * dwell, N, C)
        N -= board

    N_history.append(N)

    # 解消判定
    if clear_time is None and N <= N_safe:
        clear_time = t

# -----------------------
# 結果表示
# -----------------------
time = np.arange(T)

plt.figure()
plt.plot(time, N_history)
plt.axhline(N_safe, linestyle="--")
plt.axhline(N_th, linestyle=":")
plt.xlabel("Time [min]")
plt.ylabel("Number of passengers")
plt.title("Clearing initial congestion")
plt.show()

print(f"混雑解消時間: {clear_time} 分")
print(f"累積追加遅延: {delay_sum} 分")
