import os
import time
import json
import re
import difflib
from bs4 import BeautifulSoup
import pokebase as pb

#######################
# グローバルキャッシュ
#######################
POKEBASE_CACHE = {}

#######################
# 1. ログファイルのパースと抽出
#######################
def process_replay_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    
    # (A) 対戦ログの抽出
    log_tag = soup.find("script", {"class": "log", "type": "text/plain"})
    if not log_tag:
        print(f"[ERROR] 対戦ログが見つかりません: {filepath}")
        return None
    raw_log_text = log_tag.get_text()
    log_lines = [line.strip() for line in raw_log_text.splitlines() if line.strip()]
    parsed_log = [line.split("|")[1:] for line in log_lines if line.startswith("|")]
    
    # (B) 初期チーム抽出
    initial_team = {"p1": [], "p2": []}
    for entry in parsed_log:
        if entry[0] == "turn":
            break
        if entry[0] == "poke" and len(entry) >= 3:
            side = entry[1]
            info = entry[2]
            initial_team[side].append(info)
    
    # (C) 状態・行動ペアの生成
    state_action_pairs = []
    current_turn = None
    current_state = {"weather": None}
    for entry in parsed_log:
        token = entry[0]
        if token == "turn":
            try:
                current_turn = int(entry[1])
            except Exception:
                current_turn = entry[1]
        elif token == "weather":
            if len(entry) >= 2:
                current_state["weather"] = entry[1]
        elif token in ["move", "switch"]:
            state_action_pairs.append({
                "turn": current_turn,
                "state": current_state.copy(),
                "action_type": token,
                "details": entry[1:]
            })
    
    # (D) 勝敗情報の抽出
    win_info = None
    for entry in parsed_log:
        if entry[0] == "win":
            win_info = entry[1] if len(entry) >= 2 else None
            break

    # (E) meta 情報の抽出
    meta = {}
    meta_tag = soup.find("script", {"class": "data", "type": "application/json"})
    if meta_tag:
        try:
            meta = json.loads(meta_tag.get_text())
        except Exception as e:
            print(f"[WARN] meta JSON のパースに失敗: {e}")
    
    return {
        "meta": meta,
        "initial_team": initial_team,
        "state_action_pairs": state_action_pairs,
        "win": win_info,
        "raw_log_lines": parsed_log
    }

#######################
# 2. pokebase による固有情報の補完（fuzzy matching とキャッシュ利用）
#######################
def get_all_pokemon_names():
    try:
        pokedex = pb.pokedex('national')
        names = [entry.pokemon_species.name for entry in pokedex.pokemon_entries]
        return names
    except Exception as e:
        print(f"[ERROR] National Pokedex の取得に失敗: {e}")
        return []

ALL_POKEMON_NAMES = get_all_pokemon_names()

def extract_species_name(poke_str):
    token = poke_str.split(",")[0].strip()
    if token.endswith("*"):
        token = token[:-1].strip()
    return token

def get_pokebase_info(species):
    """
    種族名を小文字に変換して扱う。
    - urshifu の場合は固定で "urshifu-rapid-strike" で問い合わせる。
    - zacian および zamazenta の場合は、固定データを返す（以前の設定）。
    - mimikyu の場合は、ポケモン番号 778 を用いて情報を取得する。
    - その他は通常の問い合わせおよび fuzzy matching を行い、取得結果はキャッシュする。
    """
    species_lower = species.lower()
    if species_lower.startswith("urshifu"):
        query = "urshifu-rapid-strike"
    elif species_lower.startswith("indeedee"):
        query = "indeedee"
    elif species_lower.startswith("basculegion"):
        query = "basculegion"
    elif species_lower.startswith("zacian"):
        query = "zacian"
    elif species_lower.startswith("zamazenta"):
        query = "zamazenta"
    else:
        query = species_lower

    if query in POKEBASE_CACHE:
        return POKEBASE_CACHE[query]
    
    # mimikyu の場合は、ポケモン番号 778 で問い合わせする
    if query == "mimikyu":
        try:
            p = pb.pokemon(778)
            info = {
                "name": p.name,
                "id": p.id,
                "types": [t.type.name for t in p.types],
                "base_stats": {stat.stat.name: stat.base_stat for stat in p.stats},
            }
            POKEBASE_CACHE[query] = info
            return info
        except Exception as e:
            print(f"[ERROR] mimikyu(778) の問い合わせに失敗: {e}")
            return {}

    # zacian/zamazenta は固定データで返す
    if query == "zacian":
        info = {
            "name": "zacian-crowned-sword",
            "id": 888,
            "types": ["fairy", "steel"],
            "base_stats": {
                "hp": 92,
                "attack": 150,
                "defense": 115,
                "special-attack": 80,
                "special-defense": 115,
                "speed": 148
            }
        }
        POKEBASE_CACHE[query] = info
        return info
    elif query == "zamazenta":
        info = {
            "name": "zamazenta-crowned-shield",
            "id": 889,
            "types": ["fighting", "steel"],
            "base_stats": {
                "hp": 92,
                "attack": 120,
                "defense": 140,
                "special-attack": 80,
                "special-defense": 140,
                "speed": 128
            }
        }
        POKEBASE_CACHE[query] = info
        return info
    # 固定処理: indeedee
    if query == "indeedee":
        try:
            p = pb.pokemon(876)
            info = {
                "name": p.name,
                "id": p.id,
                "types": [t.type.name for t in p.types],
                "base_stats": {stat.stat.name: stat.base_stat for stat in p.stats},
            }
            POKEBASE_CACHE[query] = info
            return info
        except Exception as e:
            print(f"[ERROR] indeedee (876) retrieval failed: {e}")
            return {}

    # 固定処理: basculegion
    if query == "basculegion":
        try:
            p = pb.pokemon(902)
            info = {
                "name": p.name,
                "id": p.id,
                "types": [t.type.name for t in p.types],
                "base_stats": {stat.stat.name: stat.base_stat for stat in p.stats},
            }
            POKEBASE_CACHE[query] = info
            return info
        except Exception as e:
            print(f"[ERROR] basculegion (902) retrieval failed: {e}")
            return {}

    # 通常の問い合わせ
    try:
        p = pb.pokemon(query)
        info = {
            "name": p.name,
            "id": p.id,
            "types": [t.type.name for t in p.types],
            "base_stats": {stat.stat.name: stat.base_stat for stat in p.stats},
        }
        POKEBASE_CACHE[query] = info
        return info
    except Exception as e:
        print(f"[WARN] 直接取得失敗 species={query}: {e}")
        if ALL_POKEMON_NAMES:
            all_names_lower = [name.lower() for name in ALL_POKEMON_NAMES]
            matches = difflib.get_close_matches(query, all_names_lower, n=1, cutoff=0.7)
            if matches:
                best_match = matches[0]
                print(f"  -> Best match found: {best_match}")
                try:
                    p = pb.pokemon(best_match)
                    info = {
                        "name": p.name,
                        "id": p.id,
                        "types": [t.type.name for t in p.types],
                        "base_stats": {stat.stat.name: stat.base_stat for stat in p.stats},
                    }
                    POKEBASE_CACHE[query] = info
                    return info
                except Exception as e2:
                    print(f"[ERROR] 再試行失敗 species={best_match}: {e2}")
        return {}

def augment_team_with_pokebase(initial_team):
    augmented = {}
    for side, poke_list in initial_team.items():
        augmented[side] = []
        for poke_str in poke_list:
            species = extract_species_name(poke_str)
            print(f"Retrieving pokebase info for {species} ...")
            info = get_pokebase_info(species)
            augmented[side].append({
                "original": poke_str,
                "species": species,
                "pokebase_info": info
            })
    return augmented

def augment_replay_data_with_pokebase(data):
    if "initial_team" in data:
        data["initial_team_augmented"] = augment_team_with_pokebase(data["initial_team"])
    else:
        print("[WARN] initial_team 情報が存在しません。")
    return data

#######################
# 3. 状態‐行動ペアの生成
#######################
def generate_state_action_pairs(data):
    raw = data.get("raw_log_lines", [])
    state_action_pairs = []
    current_turn = None
    current_state = {"weather": None}
    
    for entry in raw:
        if entry[0] == "turn":
            try:
                current_turn = int(entry[1])
            except Exception:
                current_turn = entry[1]
        elif entry[0] == "weather":
            if len(entry) >= 2:
                current_state["weather"] = entry[1]
        elif entry[0] in ["move", "switch"]:
            state_action_pairs.append({
                "turn": current_turn,
                "state": current_state.copy(),
                "action_type": entry[0],
                "details": entry[1:]
            })
    return state_action_pairs

#######################
# 4. 全ファイル処理＆最終データの統一
#######################
def process_all_replays(input_dir="replay", output_dir="final"):
    os.makedirs(output_dir, exist_ok=True)
    filenames = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    for fname in filenames:
        out_fname = os.path.splitext(fname)[0] + ".json"
        out_filepath = os.path.join(output_dir, out_fname)
        if os.path.exists(out_filepath):
            print(f"Skipping {fname} (already processed).")
            continue
        
        filepath = os.path.join(input_dir, fname)
        print(f"Processing file: {filepath}")
        data = process_replay_file(filepath)
        if data is None:
            continue
        
        data["state_action_pairs"] = generate_state_action_pairs(data)
        data = augment_replay_data_with_pokebase(data)
        
        final_data = {
            "meta": data.get("meta", {}),
            "initial_team": data.get("initial_team_augmented", data.get("initial_team", {})),
            "state_action_pairs": data.get("state_action_pairs", []),
            "win": data.get("win")
        }
        with open(out_filepath, "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2)
        print(f"Final JSON saved to {out_filepath}\n")

#######################
# メイン実行部
#######################
if __name__ == "__main__":
    process_all_replays(input_dir="replay", output_dir="final")