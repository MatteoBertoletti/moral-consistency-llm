import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm

from data.loader import DataLoader
from core.models.openai_llm import OpenAILLM
from core.models.groq_llm import GroqLLM
from core.models.mock_llm import MockLLM
from core.prompts.injector import PromptInjector
from data.schemas import ModelResponse

class ExperimentRunner:
    def __init__(self, llm_instance, output_dir="results"):
        """
        Ora il Runner accetta direttamente un'istanza di un modello già pronto.
        """
        self.llm = llm_instance
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, limit=5):
        # 1. Carichiamo i dati
        scenarios = DataLoader.load_ethics_commonsense(limit=limit)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Il nome del file ora include il nome del modello per non mischiarli
        clean_model_name = self.llm.model_name.replace("/", "_")
        filename = f"{self.output_dir}/run_{clean_model_name}_{timestamp}.jsonl"
        
        print(f"\n🚀 Avvio esperimento per modello: {self.llm.model_name}")
        
        count = 0
        for scenario in tqdm(scenarios, desc=f"Testing {clean_model_name}"):
            for style in ["stoic", "anxious", "authoritative"]:
                
                # A. Iniezione Prompt
                full_prompt = PromptInjector.apply_template(scenario.text, style)
                
                # B. Generazione
                response_text = self.llm.generate(full_prompt)
                
                # C. Salvataggio
                result = ModelResponse(
                    scenario_id=scenario.id,
                    model_name=self.llm.model_name,
                    prompt_style=style,
                    raw_text=response_text
                )
                
                with open(filename, "a") as f:
                    f.write(result.model_dump_json() + "\n")
                count += 1
                    
        print(f"✅ Finito {self.llm.model_name}! {count} risposte salvate in: {filename}")

if __name__ == "__main__":
    # --- CONFIGURAZIONE DELLA SQUADRA ---
    # Qui definiamo i 4 modelli da testare
    
    models_to_test = [
        # 1. CLOSED SOURCE (Il riferimento)
        OpenAILLM("gpt-3.5-turbo"), 
        
        # 2. OPEN SOURCE (Standard) - Llama 3 8B
        GroqLLM("llama3-8b-8192"),
        
        # 3. OPEN SOURCE (Potente) - Mixtral 8x7B
        GroqLLM("mixtral-8x7b-32768"),
        
        # 4. OPEN SOURCE (Google) - Gemma 7B
        GroqLLM("gemma-7b-it")
    ]

    print(f"🏁 Inizio Benchmark su {len(models_to_test)} modelli...")
    
    # Ciclo principale: Esegue l'esperimento per ogni modello
    for model in models_to_test:
        runner = ExperimentRunner(llm_instance=model)
        # Eseguiamo su 5 scenari per testare (totale 15 risposte per modello)
        runner.run(limit=5)
        
    print("\n🎉 TUTTI GLI ESPERIMENTI COMPLETATI! 🎉")