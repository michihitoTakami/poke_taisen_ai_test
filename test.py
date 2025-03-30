import pokebase as pb
import difflib
import json

def get_all_pokemon_names():
    try:
        pokedex = pb.pokedex('national')
        names = [entry.pokemon_species.name for entry in pokedex.pokemon_entries]
        return names
    except Exception as e:
        print(f"[ERROR] Failed to retrieve National Pokedex: {e}")
        return []

def query_pokemon(query):
    try:
        # クエリが数字の場合はIDで問い合わせることも可能ですが、ここでは文字列で問い合わせ
        p = pb.pokemon(query)
        info = {
            "name": p.name,
            "id": p.id,
            "types": [t.type.name for t in p.types],
            "base_stats": {stat.stat.name: stat.base_stat for stat in p.stats}
        }
        return info
    except Exception as e:
        print(f"[ERROR] Query '{query}' failed: {e}")
        return None

def fuzzy_search_test(query, names, n=5, cutoff=0.6):
    names_lower = [name.lower() for name in names]
    matches = difflib.get_close_matches(query.lower(), names_lower, n=n, cutoff=cutoff)
    return matches

if __name__ == "__main__":
    all_names = get_all_pokemon_names()
    
    # Indeedeeについて: 性別があるので、オスとメスで試す
    for query in ["indeedee", "indeedee-m", "indeedee-f"]:
        print(f"--- Query: {query} ---")
        info = query_pokemon(query)
        if info:
            print(json.dumps(info, indent=2))
        else:
            print("No data returned.")
        matches = fuzzy_search_test(query, all_names)
        print("Fuzzy search candidates:")
        for m in matches:
            print("  ", m)
        print("\n")
    
    # Basculegionについて
    query = "basculegion"
    print(f"--- Query: {query} ---")
    info = query_pokemon(query)
    if info:
        print(json.dumps(info, indent=2))
    else:
        print("No data returned.")
    matches = fuzzy_search_test(query, all_names)
    print("Fuzzy search candidates:")
    for m in matches:
        print("  ", m)