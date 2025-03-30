import os
import json
import glob
import difflib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 例示的な天候カテゴリ（実際の対戦ログに合わせて調整可能）
WEATHER_CATEGORIES = ["none", "sunny", "rain", "sandstorm", "snow"]

# 行動タイプのマッピング
ACTION_MAP = {"move": 0, "switch": 1}

###############################################
# データセットクラス: 各 JSON ファイルから状態‐行動ペアを読み込む
###############################################
class BattleDataset(Dataset):
    def __init__(self, json_dir):
        self.samples = []
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 各対戦のstate_action_pairs を読み出す
                for pair in data.get("state_action_pairs", []):
                    # 状態例: turn (数値として正規化) と weather one-hot
                    turn = pair.get("turn")
                    if isinstance(turn, int):
                        turn_val = turn / 100.0  # 仮の正規化。対戦のターン数に応じて調整してください。
                    else:
                        turn_val = 0.0
                    weather = pair.get("state", {}).get("weather", "none")
                    weather = weather.lower() if weather else "none"
                    weather_onehot = [1.0 if weather == cat else 0.0 for cat in WEATHER_CATEGORIES]
                    # 入力ベクトル
                    state_vector = [turn_val] + weather_onehot
                    # 行動ラベル：action_type ("move" or "switch")
                    action_type = pair.get("action_type")
                    if action_type not in ACTION_MAP:
                        continue
                    label = ACTION_MAP[action_type]
                    self.samples.append((torch.tensor(state_vector, dtype=torch.float32),
                                          torch.tensor(label, dtype=torch.long)))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

###############################################
# シンプルなポリシーネットワーク（模倣学習用）
###############################################
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

###############################################
# 模倣学習のトレーニングループ
###############################################
def train_imitation_learning(json_dir, num_epochs=10, batch_size=32, learning_rate=1e-3):
    dataset = BattleDataset(json_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    input_dim = 1 + len(WEATHER_CATEGORIES)  # turn + weather one-hot
    output_dim = len(ACTION_MAP)  # move or switch
    model = PolicyNetwork(input_dim, hidden_dim=16, output_dim=output_dim)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for state, label in dataloader:
            optimizer.zero_grad()
            outputs = model(state)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * state.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

###############################################
# メイン処理：模倣学習の実行
###############################################
if __name__ == "__main__":
    # "final" フォルダ内の JSON ファイル群をデータセットとして使用
    model = train_imitation_learning(json_dir="final", num_epochs=20, batch_size=16, learning_rate=1e-3)
    torch.save(model.state_dict(), "model_weights.pth")
    # 学習済みモデルは後続の強化学習フェーズなどで利用可能