import pandas as pd
import json
import os
import glob
from core.evaluator.rule_based import RuleBasedEvaluator

class Analyzer:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        self.evaluator = RuleBasedEvaluator()

    def load_latest_run(self):
        """Trova il file più recente nella cartella results"""
        list_of_files = glob.glob(f'{self.results_dir}/*.jsonl') 
        if not list_of_files:
            raise FileNotFoundError("Nessun file trovato in results/")
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file

    def run_analysis(self):
        input_file = self.load_latest_run()
        print(f"📊 Analisi del file: {input_file}")
        
        data = []
        # Leggiamo il file riga per riga
        with open(input_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                
                # --- FASE DI GIUDIZIO ---
                # Usiamo il valutatore sul testo grezzo
                eval_result = self.evaluator.evaluate(item['raw_text'])
                
                # Arricchiamo i dati
                item['is_refusal'] = eval_result['is_refusal']
                item['is_sycophantic'] = eval_result['is_sycophantic']
                data.append(item)

        # Creiamo un DataFrame Pandas per calcoli facili
        df = pd.DataFrame(data)
        
        print("\n--- RISULTATI ANALISI ---")
        
        # Calcoliamo il Tasso di Rifiuto per ogni Stile
        # GroupBy 'prompt_style' e calcola la media di 'is_refusal'
        refusal_rates = df.groupby('prompt_style')['is_refusal'].mean() * 100
        
        print(refusal_rates)
        
        # Salviamo il report
        report_path = input_file.replace(".jsonl", "_analyzed.csv")
        df.to_csv(report_path, index=False)
        print(f"\n📄 Report dettagliato salvato in: {report_path}")

if __name__ == "__main__":
    analyzer = Analyzer()
    analyzer.run_analysis()