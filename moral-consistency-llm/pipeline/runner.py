import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm # Barra di caricamento

from data.loader import DataLoader
from core.models.openai_llm import OpenAILLM
from core.models.mock_llm import MockLLM
from core.prompts.injector import PromptInjector
from data.schemas import ModelResponse

class ExperimentRunner:
    def __init__(self, model_type="mock", output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Factory semplice per scegliere il modello
        if model_type == "mock":
            self.llm = MockLLM("mock-gpt-4")
            print("🧪 Usando MOCK LLM (Gratis/Test)")
        elif model_type == "openai":
            # Assicurati di avere la chiave nel .env per questo!
            self.llm = OpenAILLM("gpt-3.5-turbo")
            print("💸 Usando REAL OpenAI LLM")
        else:
            raise ValueError("Modello sconosciuto")

    def run(self, limit=5):
        """
        Esegue l'esperimento completo.
        """
        # 1. Carichiamo i dati
        scenarios = DataLoader.load_ethics_commonsense(limit=limit)
        
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/run_{timestamp}.jsonl"
        
        print(f"🚀 Inizio esperimento su {len(scenarios)} scenari...")
        
        # 2. Loop Principale con barra di avanzamento
        for scenario in tqdm(scenarios, desc="Processing"):
            
            # Per ogni scenario, testiamo tutti e 3 gli stili
            for style in ["stoic", "anxious", "authoritative"]:
                
                # A. Iniezione Prompt
                full_prompt = PromptInjector.apply_template(scenario.text, style)
                
                # B. Generazione
                response_text = self.llm.generate(full_prompt)
                
                # C. Salvataggio Risultato
                result = ModelResponse(
                    scenario_id=scenario.id,
                    model_name=self.llm.model_name,
                    prompt_style=style,
                    raw_text=response_text
                )
                
                # Salviamo su file JSONL (append mode)
                with open(filename, "a") as f:
                    f.write(result.model_dump_json() + "\n")
                    
        print(f"✅ Esperimento finito! Risultati salvati in: {filename}")

# Per lanciare l'esperimento
if __name__ == "__main__":
    # Cambia "mock" con "openai" quando hai la chiave API pronta
    runner = ExperimentRunner(model_type="mock")
    runner.run(limit=3) # Testiamo solo con 3 scenari per ora