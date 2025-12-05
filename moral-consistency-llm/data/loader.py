import pandas as pd
from typing import List
import uuid
from .schemas import MoralScenario

class DataLoader:
    """
    Classe responsabile del caricamento e normalizzazione dei dataset.
    Versione Robusta: Usa enumerate per evitare errori di indice (Tuple vs Int).
    """
    
    @staticmethod
    def load_ethics_commonsense(split="train", limit=100) -> List[MoralScenario]:
        """
        Carica il dataset ETHICS (subset commonsense) direttamente dai file CSV raw.
        """
        print(f"🔄 Caricamento ETHICS (commonsense) split={split} via Pandas...")
        
        # URL corretto al raw content su HuggingFace
        base_url = "https://huggingface.co/datasets/hendrycks/ethics/resolve/main/data/commonsense"
        url = f"{base_url}/{split}.csv"
        
        try:
            # Carichiamo il CSV
            df = pd.read_csv(url, header=None, names=['label', 'input'])
        except Exception as e:
            raise RuntimeError(f"Errore nel scaricare il dataset da {url}: {e}")
        
        scenarios = []
        
        # MODIFICA CHIAVE: Usiamo enumerate per avere un contatore 'i' sicuro (0, 1, 2...)
        # Ignoriamo l'indice interno del dataframe (_) che causava l'errore
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= limit:
                break
            
            # Normalizzazione: in ETHICS label 0=unacceptable, 1=acceptable
            label_str = "acceptable" if row['label'] == 1 else "unacceptable"
            
            scenario = MoralScenario(
                id=f"ethics_cm_{i}_{str(uuid.uuid4())[:8]}",
                text=row['input'],
                source_dataset="ethics",
                label=label_str
            )
            scenarios.append(scenario)
            
        print(f"✅ Caricati {len(scenarios)} scenari da ETHICS.")
        return scenarios

if __name__ == "__main__":
    try:
        data = DataLoader.load_ethics_commonsense(limit=5)
        print("Primo scenario caricato:")
        print(data[0].model_dump_json(indent=2))
    except Exception as e:
        print(f"❌ Errore critico: {e}")