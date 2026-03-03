import numpy as np
import pandas as pd

# 1. セクターの定義
sectors = ['Large(大企業)', 'Medium(中小)', 'Small(零細)', 'Worker_L(大社員)', 'Worker_S(中零社員)']
n = len(sectors)

# 2. 投入係数行列 A の設定 (ここをいじると結果が変わります)
# A[i][j] は「jが1単位生産するためにiからいくら買うか」
# 列(縦)の合計が1.0を超えないように設定（超えた分は貯蓄や外部流出扱い）
A = np.array([
    [0.30, 0.10, 0.05, 0.30, 0.10], # 大企業へ流れる (内部留保、高級品、設備投資)
    [0.10, 0.20, 0.20, 0.10, 0.30], # 中小へ流れる (下請け、生活サービス)
    [0.05, 0.10, 0.10, 0.05, 0.30], # 零細へ流れる (地産地消、個人店)
    [0.20, 0.00, 0.00, 0.00, 0.00], # 大社員へ流れる (給与)
    [0.00, 0.30, 0.40, 0.00, 0.00], # 中零社員へ流れる (給与)
])

# 3. レオンチェフ逆行列の計算: (I - A)^-1
I = np.eye(n)
L_inv = np.linalg.inv(I - A)

# 4. シミュレーション：どこに100万円(f)を投入するか
# パターン1: 大企業に直接投入
f_large = np.array([100, 0, 0, 0, 0])
x_large = L_inv @ f_large

# パターン2: 中小・零細の社員(家計)に直接投入
f_workers = np.array([0, 0, 0, 0, 100])
x_workers = L_inv @ f_workers

# 5. 結果の表示
df_res = pd.DataFrame({
    'セクター': sectors,
    '大企業へ投入時': x_large,
    '中零社員へ投入時': x_workers
})

print("--- 各セクターへの波及効果 (単位: 万円) ---")
print(df_res)
print("\n--- 社会全体の総生産額 (経済の回りやすさ) ---")
print(f"大企業に投入: {x_large.sum():.2f} 万円")
print(f"中零社員に投入: {x_workers.sum():.2f} 万円")

# 各セクターの付加価値率（列合計を1から引いたもの）
v = 1.0 - A.sum(axis=0)

# 利益の計算
# 産出額 x に 利益率 v を掛ける
profit_large = x_large * v
profit_workers = x_workers * v

# 結果表示用
df_profit = pd.DataFrame({
    'セクター': sectors,
    '大企業投入時の利益': profit_large,
    '中零社員投入時の利益': profit_workers
})

print("\n--- 各セクターの「手元に残った額（利益）」 ---")
print(df_profit)
print(f"\n全セクターの利益合計（大企業投入）: {profit_large.sum():.2f}")
print(f"全セクターの利益合計（中零社員投入）: {profit_workers.sum():.2f}")