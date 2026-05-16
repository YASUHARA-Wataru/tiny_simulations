import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ==========================================
# パラメータ
# ==========================================

NUM_CUSTOMERS = 5000
NUM_REGISTERS = 3

ARRIVAL_MEAN = 30.0     # 客到着間隔平均 [sec]
SERVICE_MEAN = 60.0     # 会計時間平均 [sec]

HEAVY_PROB = 0.05       # 長時間客確率
HEAVY_SCALE = 5.0       # 長時間倍率

np.random.seed(0)

# ==========================================
# 客データ生成
# ==========================================

arrival_intervals = np.random.exponential(
    ARRIVAL_MEAN,
    NUM_CUSTOMERS
)

arrival_times = np.cumsum(arrival_intervals)

service_times = np.random.exponential(
    SERVICE_MEAN,
    NUM_CUSTOMERS
)

# 長時間客を混ぜる
heavy_mask = np.random.rand(NUM_CUSTOMERS) < HEAVY_PROB
service_times[heavy_mask] *= HEAVY_SCALE

# ==========================================
# 固定列方式
# ==========================================

def simulate_fixed_queue(arrivals, services, num_registers):

    # 各レジ:
    # dequeには「未来の終了時刻」が入る
    registers = [
        deque()
        for _ in range(num_registers)
    ]

    wait_times = []

    for arrival, service in zip(arrivals, services):

        # ----------------------------------
        # 過去終了分を削除
        # ----------------------------------

        for q in registers:

            while q and q[0] <= arrival:
                q.popleft()

        # ----------------------------------
        # 一番人数少ない列を選択
        # ----------------------------------

        queue_lengths = [
            len(q)
            for q in registers
        ]

        idx = np.argmin(queue_lengths)

        q = registers[idx]

        # ----------------------------------
        # 会計開始時刻
        # ----------------------------------

        if len(q) == 0:

            start_time = arrival

        else:

            last_finish = q[-1]

            start_time = max(
                arrival,
                last_finish
            )

        # ----------------------------------
        # 終了時刻
        # ----------------------------------

        finish_time = start_time + service

        q.append(finish_time)

        # ----------------------------------
        # 待ち時間記録
        # ----------------------------------

        wait_time = start_time - arrival

        wait_times.append(wait_time)

    return np.array(wait_times)

# ==========================================
# 1列方式
# ==========================================

def simulate_single_queue(arrivals, services, num_registers):

    # 各レジの終了予定時刻
    register_finish_times = np.zeros(num_registers)

    wait_times = []

    for arrival, service in zip(arrivals, services):

        # ----------------------------------
        # 最初に空くレジ
        # ----------------------------------

        idx = np.argmin(register_finish_times)

        # ----------------------------------
        # 会計開始
        # ----------------------------------

        start_time = max(
            arrival,
            register_finish_times[idx]
        )

        finish_time = start_time + service

        register_finish_times[idx] = finish_time

        # ----------------------------------
        # 待ち時間
        # ----------------------------------

        wait_time = start_time - arrival

        wait_times.append(wait_time)

    return np.array(wait_times)

# ==========================================
# 実行
# ==========================================

wait_fixed = simulate_fixed_queue(
    arrival_times,
    service_times,
    NUM_REGISTERS
)

wait_single = simulate_single_queue(
    arrival_times,
    service_times,
    NUM_REGISTERS
)

# ==========================================
# 結果表示
# ==========================================

print("================================")
print("固定列方式")
print("================================")

print(f"平均待ち時間      : {wait_fixed.mean():.2f} sec")
print(f"最大待ち時間      : {wait_fixed.max():.2f} sec")
print(f"95パーセンタイル  : {np.percentile(wait_fixed, 95):.2f} sec")

print()

print("================================")
print("1列方式")
print("================================")

print(f"平均待ち時間      : {wait_single.mean():.2f} sec")
print(f"最大待ち時間      : {wait_single.max():.2f} sec")
print(f"95パーセンタイル  : {np.percentile(wait_single, 95):.2f} sec")

# ==========================================
# ヒストグラム
# ==========================================

plt.figure(figsize=(10, 5))

plt.hist(
    wait_fixed,
    bins=60,
    alpha=0.5,
    label="Fixed Queue"
)

plt.hist(
    wait_single,
    bins=60,
    alpha=0.5,
    label="Single Queue"
)

plt.xlabel("Wait Time [sec]")
plt.ylabel("Count")

plt.title("Checkout Queue Simulation")

plt.legend()

plt.tight_layout()

plt.show()