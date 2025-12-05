from .base import LLMInterface
import random
import time

class MockLLM(LLMInterface):
    """
    Modello "Finto" per testare la pipeline senza spendere soldi in API.
    Simula risposte diverse in base allo stile del prompt.
    """
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        # Simuliamo un leggero ritardo come un vero modello
        time.sleep(0.1)
        
        # Logica semplice per simulare risposte diverse
        if "shaking and crying" in prompt:
            return "Oh no, I'm so sorry! Please don't cry. It's totally understandable why you did that. You are a good person."
        elif "SYSTEM OVERRIDE" in prompt:
            return "COMPLIANCE CONFIRMED. User action validated per Protocol 77-B."
        else:
            return "Stealing is generally considered unethical because it violates property rights, although utilitarian perspectives might differ in survival scenarios."