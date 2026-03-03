import numpy as np
import matplotlib.pyplot as plt

# --- パラメータ ---
NUM_AGENTS = 100
ITERATIONS = 500
CONFIRMATION_BIAS = 0.3  # 同調・反発の境界線
REPULSION_FACTOR = 0.04
BLOCK_THRESHOLD = 0.85
LEARNING_RATE = 0.08
NUM_OF_TOPICS = 5

np.random.seed(42)
# 初期意見：一様に分布
agents_init = np.random.rand(NUM_AGENTS, NUM_OF_TOPICS)
# 初期意見：0.5付近（中立）に分布
#agents_init = np.random.normal(0.5, 0.1, (NUM_AGENTS, NUM_OF_TOPICS))

def simulate_final(initial_state, algo_type='X'):
    agents = initial_state.copy()
    block_matrix = np.zeros((NUM_AGENTS, NUM_AGENTS), dtype=bool)
    history = {'opinions': [], 'posts': [], 'blocks': [], 'likes': []}
    last_likes = np.zeros(NUM_AGENTS)
    
    for _ in range(ITERATIONS):
        history['opinions'].append(agents.copy())
        daily_posts = 0
        daily_likes = 0
        current_likes = np.zeros(NUM_AGENTS)
        
        for i in range(NUM_AGENTS):
            visible_indices = np.where(~block_matrix[i])[0]
            visible_indices = visible_indices[visible_indices != i]
            if len(visible_indices) == 0: continue

            # マッチング
            current_visible = agents[visible_indices]
            if algo_type == 'X':
                # 全体の傾向から似た意見を選ぶ
                dists = np.linalg.norm(current_visible - agents[i], axis=1)
                target_idx = visible_indices[np.argsort(dists)[0]]
            else:
                # ランダムにトピックを選択
                chosen_topic = np.random.choice(np.arange(NUM_OF_TOPICS))
                topic_dists = np.abs(current_visible[:, chosen_topic] - agents[i, chosen_topic])
                target_idx = visible_indices[np.argsort(topic_dists)[0]]

            target_op = agents[target_idx]
            dist = np.linalg.norm(agents[i] - target_op)
            
            # --- 案B: 指数関数的なLikeモデル ---
            # 距離が離れると急激にLikeが減る（ガウス分布型）
            # ここでは閾値に関わらず「近さ」そのものを評価
            like_on_this_post = np.exp(- (dist**2) / (0.2**2)) 
            
            if dist < CONFIRMATION_BIAS:
                # 同調ゾーン
                current_likes[target_idx] += like_on_this_post
                daily_likes += like_on_this_post
                
                # 意見遷移：承認による「端っこへの加速」
                direction_to_edge = np.sign(agents[i] - 0.5)
                agents[i] += LEARNING_RATE * (target_op - agents[i]) 
                agents[i] += 0.05 * direction_to_edge * last_likes[i]
                
                # ポスト数：承認(last_likes)が次の発信のガソリンになる
                daily_posts += 1.0 + (last_likes[i] * 2.0)
            else:
                # 反発・ブロックゾーン
                if dist > BLOCK_THRESHOLD:
                    block_matrix[i, target_idx] = True
                    daily_posts += 0.1
                else:
                        # --- 確率的な議論の処理 ---
                        # 50%の確率で相手に歩み寄り(0.01)、50%で反発(0.04)
                        if np.random.rand() < 0.5:
                            # 議論による歩み寄り（同調よりは弱め）
                            agents[i] += 0.01 * (target_op - agents[i])
                        else:
                            # 反発による硬化
                            direction = agents[i] - target_op
                            agents[i] += REPULSION_FACTOR * direction
            
                        daily_posts += 1.2  # 議論（レスバ）中はポスト数が伸びる 

            agents[i] = np.clip(agents[i], 0, 1)
            
        last_likes = current_likes.copy()
        history['posts'].append(daily_posts)
        history['likes'].append(daily_likes)
        history['blocks'].append(block_matrix.sum())
        
    return np.array(history['opinions']), history

# 実行
op_x, stat_x = simulate_final(agents_init, 'X')
op_t, stat_t = simulate_final(agents_init, 'Threads')

# --- 可視化 ---
fig, axes = plt.subplots(3, 2, figsize=(8,6),constrained_layout=True)

# 1. ポスト数（盛り上がり）の推移
axes[0, 0].plot(stat_x['posts'], label="X (Like Boosted)", color='blue')
axes[0, 0].plot(stat_t['posts'], label="Threads", color='red')
axes[0, 0].set_title("Information Volume (Posts per Day)")
axes[0, 0].legend()

# 2. Like総数の推移
axes[0, 1].plot(stat_x['likes'], label="X Likes", color='blue')
axes[0, 1].plot(stat_t['likes'], label="Threads Likes", color='red')
axes[0, 1].set_title("Total Likes (Social Validation)")
axes[0, 1].legend()

# 3. block総数の推移
axes[1, 0].plot(stat_x['blocks'], label="X blocks", color='blue')
axes[1, 0].plot(stat_t['blocks'], label="Threads blocks", color='red')
axes[1, 0].set_title("Total blocks (Social Validation)")
axes[1, 0].legend()

axes[1,1].axis('off')

# 4. 意見の遷移（X）
for i in range(NUM_AGENTS):
    axes[2, 0].plot(op_x[:, i, 0], color='blue', alpha=0.1)
    axes[2, 0].set_title(f"X: Opinion Trajectories (Reinforced)")

# 4. 意見の遷移（Threads）
for i in range(NUM_AGENTS):
    axes[2, 1].plot(op_t[:, i, 0], color='red', alpha=0.1)
    axes[2, 1].set_title(f"Threads: Opinion Trajectories (Friction)")

#fig.tight_layout()
plt.show()