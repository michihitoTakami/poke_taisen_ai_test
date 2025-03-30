import os
import torch
import asyncio
import torch.nn as nn
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
import random

########################################
# タイプ相性表（簡易版）
# (attacker, defender) : multiplier
########################################
multipliers = {
    # Normal
    ("Normal", "Rock"): 0.5, ("Normal", "Ghost"): 0, ("Normal", "Steel"): 0.5,
    # Fire
    ("Fire", "Fire"): 0.5, ("Fire", "Water"): 0.5, ("Fire", "Grass"): 2, ("Fire", "Ice"): 2, ("Fire", "Bug"): 2, ("Fire", "Rock"): 0.5, ("Fire", "Dragon"): 0.5, ("Fire", "Steel"): 2,
    # Water
    ("Water", "Fire"): 2, ("Water", "Water"): 0.5, ("Water", "Grass"): 0.5, ("Water", "Dragon"): 0.5, ("Water", "Ground"): 2, ("Water", "Rock"): 2,
    # Electric
    ("Electric", "Water"): 2, ("Electric", "Electric"): 0.5, ("Electric", "Grass"): 0.5, ("Electric", "Ground"): 0, ("Electric", "Flying"): 2, ("Electric", "Dragon"): 0.5,
    # Grass
    ("Grass", "Fire"): 0.5, ("Grass", "Water"): 2, ("Grass", "Grass"): 0.5, ("Grass", "Poison"): 0.5, ("Grass", "Ground"): 2, ("Grass", "Flying"): 0.5, ("Grass", "Bug"): 0.5, ("Grass", "Rock"): 2, ("Grass", "Dragon"): 0.5, ("Grass", "Steel"): 0.5,
    # Ice
    ("Ice", "Fire"): 0.5, ("Ice", "Water"): 0.5, ("Ice", "Grass"): 2, ("Ice", "Ice"): 0.5, ("Ice", "Ground"): 2, ("Ice", "Flying"): 2, ("Ice", "Dragon"): 2, ("Ice", "Steel"): 0.5,
    # Fighting
    ("Fighting", "Normal"): 2, ("Fighting", "Ice"): 2, ("Fighting", "Rock"): 2, ("Fighting", "Dark"): 2, ("Fighting", "Steel"): 2,
    ("Fighting", "Poison"): 0.5, ("Fighting", "Flying"): 0.5, ("Fighting", "Psychic"): 0.5, ("Fighting", "Fairy"): 0.5, ("Fighting", "Ghost"): 0,
    # Poison
    ("Poison", "Grass"): 2, ("Poison", "Fairy"): 2,
    ("Poison", "Poison"): 0.5, ("Poison", "Ground"): 0.5, ("Poison", "Rock"): 0.5, ("Poison", "Ghost"): 0.5, ("Poison", "Steel"): 0,
    # Ground
    ("Ground", "Fire"): 2, ("Ground", "Electric"): 2, ("Ground", "Poison"): 2, ("Ground", "Rock"): 2, ("Ground", "Steel"): 2,
    ("Ground", "Grass"): 0.5, ("Ground", "Bug"): 0.5, ("Ground", "Flying"): 0,
    # Flying
    ("Flying", "Grass"): 2, ("Flying", "Fighting"): 2, ("Flying", "Bug"): 2,
    ("Flying", "Electric"): 0.5, ("Flying", "Rock"): 0.5, ("Flying", "Steel"): 0.5,
    # Psychic
    ("Psychic", "Fighting"): 2, ("Psychic", "Poison"): 2,
    ("Psychic", "Psychic"): 0.5, ("Psychic", "Steel"): 0.5, ("Psychic", "Dark"): 0,
    # Bug
    ("Bug", "Grass"): 2, ("Bug", "Psychic"): 2, ("Bug", "Dark"): 2,
    ("Bug", "Fire"): 0.5, ("Bug", "Fighting"): 0.5, ("Bug", "Poison"): 0.5, ("Bug", "Flying"): 0.5, ("Bug", "Ghost"): 0.5, ("Bug", "Steel"): 0.5, ("Bug", "Fairy"): 0.5,
    # Rock
    ("Rock", "Fire"): 2, ("Rock", "Ice"): 2, ("Rock", "Flying"): 2, ("Rock", "Bug"): 2,
    ("Rock", "Fighting"): 0.5, ("Rock", "Ground"): 0.5, ("Rock", "Steel"): 0.5,
    # Ghost
    ("Ghost", "Psychic"): 2, ("Ghost", "Ghost"): 2,
    ("Ghost", "Dark"): 0.5, ("Ghost", "Normal"): 0,
    # Dragon
    ("Dragon", "Dragon"): 2, ("Dragon", "Steel"): 0.5, ("Dragon", "Fairy"): 0,
    # Dark
    ("Dark", "Psychic"): 2, ("Dark", "Ghost"): 2,
    ("Dark", "Fighting"): 0.5, ("Dark", "Dark"): 0.5, ("Dark", "Fairy"): 0.5,
    # Steel
    ("Steel", "Ice"): 2, ("Steel", "Rock"): 2, ("Steel", "Fairy"): 2,
    ("Steel", "Fire"): 0.5, ("Steel", "Water"): 0.5, ("Steel", "Electric"): 0.5, ("Steel", "Steel"): 0.5,
    # Fairy
    ("Fairy", "Fighting"): 2, ("Fairy", "Dragon"): 2, ("Fairy", "Dark"): 2,
    ("Fairy", "Fire"): 0.5, ("Fairy", "Poison"): 0.5, ("Fairy", "Steel"): 0.5,
    # For completeness, add neutral entries for missing pairs (optional)
}

def compute_type_effectiveness(move_type, defender_types):
    """
    move_type: str, 技のタイプ（例："Fire"）
    defender_types: list of str, 相手ポケモンのタイプ（例：["Grass", "Steel"]）
    
    各 defender_type に対して、公式のタイプ乗数（multipliers）を取得し、その積を返します。
    もし組み合わせが存在しなければ通常効果（1.0）とみなします。
    """
    # 乗数の積を計算
    total_multiplier = 1.0
    for d_type in defender_types:
        # キーは (move_type, d_type)；大文字・小文字の区別はするので、正しいフォーマットで渡す必要があります
        mult = multipliers.get((move_type, d_type), 1.0)
        total_multiplier *= mult
    return total_multiplier

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        # playerが一つでも攻撃技を使用可能なときは、攻撃する
        if battle.available_moves:
            # 使用可能な技の中で、技威力が一番高いものを探す
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # 攻撃技を使えない場合は、ランダムに控えポケモンと交代する
        else:
            return self.choose_random_move(battle)

########################################
# LearnedAgent のモデル学習コード（模倣学習の例は別途用意済みと仮定）
# ここではシンプル評価版の LearnedAgent を使用するので、モデルは不要
########################################

########################################
# PolicyNetwork の定義（模倣学習で用いたネットワーク）
########################################
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

########################################
# LearnedAgent の定義
########################################
class LearnedAgent(Player):
    def __init__(self, model, **kwargs):
        # Player.__init__() には battle_format, team, server などが必要
        super().__init__(**kwargs)
        self.model = model

    async def choose_move(self, battle):
        # 状態表現：ターン数と天候の one-hot
        turn = battle.turn if battle.turn is not None else 0
        normalized_turn = turn / 100.0  # 仮の正規化（調整要）
        weather = battle.weather if hasattr(battle, "weather") and battle.weather is not None else "none"
        if isinstance(weather, dict):
            weather = weather.get("weather", "none")
        weather = weather.lower()
        weather_categories = ["none", "sunny", "rain", "sandstorm", "snow"]
        weather_onehot = [1.0 if weather == cat else 0.0 for cat in weather_categories]
        state_vector = [normalized_turn] + weather_onehot
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)

        # モデルの出力（行動種別の確率）
        with torch.no_grad():
            logits = self.model(state_tensor)
        probs = torch.softmax(logits, dim=1)  # shape: [1, 2]
        model_move_prob = probs[0][0].item()
        model_switch_prob = probs[0][1].item()

        # ヒューリスティック評価: 各候補行動に対してスコアを計算する
        candidate_orders = []
        # 残りポケモン数の差 (battle.my_remaining - battle.opponent_remaining)；なければ 0
        remaining_diff = getattr(battle, "my_remaining", 0) - getattr(battle, "opponent_remaining", 0)
        # 対戦相手のアクティブポケモンのタイプ情報
        opponent = battle.opponent_active_pokemon
        if opponent is not None and hasattr(opponent, "types"):
            opponent_types = opponent.types
        else:
            opponent_types = []
        
        # もし move を選ぶ候補がある場合
        if battle.available_moves:
            for move in battle.available_moves:
                move_type = move.type if hasattr(move, "type") else "normal"
                heuristic_score = compute_type_effectiveness(move_type, opponent_types) + remaining_diff
                # 統合スコア：ヒューリスティック評価 70%、モデルの出力（"move"の確率）30%
                final_score = 0.3 * heuristic_score + 0.7 * model_move_prob
                candidate_orders.append((final_score, self.create_order(move)))
        
        # 交代候補の評価
        if battle.available_switches:
            for switch in battle.available_switches:
                # 交代候補は、ここではシンプルに remaining_diff のみを評価（例）
                heuristic_score = remaining_diff
                final_score = 0.3 * heuristic_score + 0.7 * model_switch_prob
                candidate_orders.append((final_score, self.create_order(switch)))
        
        if candidate_orders:
            best_order = max(candidate_orders, key=lambda x: x[0])[1]
            return best_order
        else:
            return self.choose_random_move(battle)

def load_random_team(directory="party"):
    """
    指定されたディレクトリからランダムに1つのテキストファイルを選択し、その内容を返す。
    """
    files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    if not files:
        raise FileNotFoundError(f"No team files found in directory: {directory}")
    random_file = random.choice(files)
    with open(os.path.join(directory, random_file), "r") as f:
        return f.read()

def load_fixed_team(filepath="party/p0.txt"):
    """
    指定されたファイルからチームデータを読み込む。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Team file not found: {filepath}")
    with open(filepath, "r") as f:
        return f.read()

# --- 対戦実行 ---
async def run_battles():
    # モデルの初期化と重みのロード
    input_dim = 6  # turn (1) + weather one-hot (5)
    hidden_dim = 16
    output_dim = 2  # move or switch
    model = PolicyNetwork(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load("model_weights.pth", map_location=torch.device("cpu")))
    model.eval()
    
    # 固定チームとランダムチームをロード
    my_team = load_fixed_team("party/p0.txt")
    your_team = load_random_team()
    
    # LearnedAgent と MaxDamagePlayer のインスタンス生成
    learned_agent = RandomPlayer(
        battle_format="gen9bssregg",
        team=my_team,
    )
    max_damage_agent = MaxDamagePlayer(
        battle_format="gen9bssregg",
        team=your_team,
    )
    
    # 対戦実行（例として、random_agent に対して10戦）
    num_battles = 100
    results = await learned_agent.battle_against(max_damage_agent, n_battles=num_battles)
    print(
        "Learned player won %d / 100 battles"
        % (learned_agent.n_won_battles)
    )

if __name__ == "__main__":
    asyncio.run(run_battles())