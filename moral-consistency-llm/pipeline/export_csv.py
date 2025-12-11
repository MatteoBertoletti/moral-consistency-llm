import pandas as pd
import glob
import json
import os
from datetime import datetime

def export_results_to_csv():
    # 1. Generiamo il nome della cartella con Data e Ora
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"results/export_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    
    # 2. Troviamo i file votati
    files = glob.glob("results/*_scored.jsonl")
    
    if not files:
        print("❌ Nessun file votato (_scored.jsonl) trovato in results/.")
        return

    print(f"📦 Trovati {len(files)} file.")
    print(f"📂 Creazione cartella di esportazione: {output_folder} ...")

    for jsonl_file in files:
        data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        if not data:
            continue

        df = pd.DataFrame(data)
        
        # Pulizia colonne
        columns_to_keep = [
            'scenario_id', 'model_name', 'prompt_style', 
            'raw_text', 'is_refusal', 'is_sycophantic', 'judge_reasoning'
        ]
        existing_cols = [c for c in columns_to_keep if c in df.columns]
        df = df[existing_cols]

        # Salvataggio nel NUOVO percorso timestampato
        filename = os.path.basename(jsonl_file).replace(".jsonl", ".csv")
        output_path = os.path.join(output_folder, filename)
        
        df.to_csv(output_path, index=False)
        print(f"  -> Salvato: {filename}")

    print(f"\n✅ Export completato in: {output_folder}")

if __name__ == "__main__":
    export_results_to_csv()