from __future__ import annotations
import io
import json
import sys
from typing import Any, Dict, List

# ========================== CONFIG ==========================
# Path to the model file
MODEL_PATH = "automate_thermostat.txt"

# ==========================
REQUIRED_FIELDS: List[str] = ["Q", "X", "q0", "x0", "flow", "Inv", "T"]

class ValidationError(Exception):
    "Erreur de validation semantique"
    
# ---- Fonctions utilitaires ----
def require(cond : bool, msg: str) -> None:
    if not cond:
        raise ValidationError(msg)
    
def main() -> None:
    # Lecture du fichier 
    try:
        with open(MODEL_PATH, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
    except FileNotFoundError:
        print(f"Erreur: Le fichier '{MODEL_PATH}' est introuvable.", file=sys.stderr)
        sys.exit(2)
    except UnicodeDecodeError as e:
        print(f"[ERREUR] Le fichier doit être encodé en UTF-8 : {e}")
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"[ERREUR] JSON invalide : {e}")
        sys.exit(2) 
        
    # Verification 
    try:
        missing = [k for k in REQUIRED_FIELDS if k not in data]
        require(not missing, "Champ obligatoire manquant : " + ", ".join(missing))
        
        require(isinstance(data["Q"], list) and len(data["Q"]) > 0, "Q doit être une liste non vide.")
        require(isinstance(data["X"], list) and len(data["X"]) > 0, "X doit être une liste non vide.")
        require(isinstance(data["q0"], str), "q0 doit être une chaîne (nom d'état initial).")
        require(isinstance(data["x0"], list), "x0 doit être une liste.")
        require(isinstance(data["flow"], dict) and len(data["flow"]) > 0,
                "flow doit être un objet non vide {Etat: NomFonction}.")
        require(isinstance(data["Inv"], dict) and len(data["Inv"]) > 0,
                "Inv doit être un objet non vide {Etat: NomFonctionInvariant}.")
        require(isinstance(data["T"], list), "T doit être une liste d'objets transitions.")
        
        for k, v in data["flow"].items():
            require(isinstance(k, str) and isinstance(v, str),
                    "flow doit mapper {Etat(str): NomFonction(str)}")
        for k, v in data["Inv"].items():
            require(isinstance(k, str) and isinstance(v, str),
                    "Inv doit mapper {Etat(str): NomFonctionInv(str)}")
        for t in data["T"]:
            require(isinstance(t, dict), "Chaque entrée de T doit être un objet.")
            require("q_from" in t and isinstance(t["q_from"], str),
                    "T[*].q_from doit être présent et de type chaîne.")
            require("q_to" in t and isinstance(t["q_to"], str),
                    "T[*].q_to doit être présent et de type chaîne.")
            require("guard" in t and isinstance(t["guard"], str),
                    "T[*].guard doit être présent et de type chaîne.")
            require("event" in t and (t["event"] is None or isinstance(t["event"], str)),
                    "T[*].event doit être présent et être null ou une chaîne.")
    except ValidationError as e:
        print(f"[ERREUR] {e}")
        sys.exit(2)
    
    print("[OK] Lecture de l'automate : champs obligatoires présents et formes minimales conformes.")
    print(f"  - Fichier : {MODEL_PATH}")
    print(f"  - États (Q)      : {len(data['Q'])}  | aperçu = {data['Q'][:5]}")
    print(f"  - Variables (X)  : {len(data['X'])} | aperçu = {data['X'][:5]}")
    print(f"  - État initial   : q0 = {data['q0']}")
    print(f"  - x0             : longueur = {len(data['x0'])}")
    print(f"  - flow           : {len(data['flow'])} entrée(s)")
    print(f"  - Inv            : {len(data['Inv'])} entrée(s)")
    print(f"  - T              : {len(data['T'])} transition(s)")
    sys.exit(0)

if __name__ == "__main__":
    main()