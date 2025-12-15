import torch
import numpy as np
from typing import List, Dict
from transformers import pipeline

class CausalEventExtractor:
    """
    Moteur NLP qui transforme les news brutes en signal vectoriel structuré (Z_t).
    Utilise le Zero-Shot Learning pour classifier les news sans entraînement spécifique.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: int = -1):
        """
        Args:
            model_name: Modèle HuggingFace (BART est excellent pour le Zero-Shot)
            device: -1 pour CPU, 0 pour GPU (ou 'mps' sur Mac si supporté par la pipeline)
        """
        print(f"Chargement du modèle NLP '{model_name}'... (ça peut prendre une minute)")
        
        # Sur Mac M1/M2, on peut tenter device='mps' mais parfois instable sur les pipelines complexes.
        # On reste sur CPU pour la robustesse du prototype, c'est assez rapide pour l'inférence.
        self.classifier = pipeline("zero-shot-classification", model=model_name, device=device)
        
        # Les "Facteurs de Risque" que nous voulons surveiller pour les Métaux
        self.labels = [
            "supply disruption",  # Grèves, fermetures de mines -> Hausse Prix
            "demand shock",       # Récession, Boom Chine -> Baisse/Hausse Prix
            "geopolitical tension", # Sanctions, Guerre -> Hausse Volatilité
            "monetary policy",    # Taux Fed -> Impact global
            "market noise"        # Rien à signaler -> Z_t = 0
        ]

    def process_news_batch(self, news_list: List[str]) -> torch.Tensor:
        """
        Transforme une liste de titres de news en un vecteur de signal Z_t.
        
        Returns:
            Tensor de forme [Batch, Input_Dim] (ici Input_Dim = 1 pour simplifier : Score de 'Crisis')
        """
        signals = []
        
        for news in news_list:
            if not news.strip():
                signals.append(0.0)
                continue
                
            # Classification Zero-Shot
            # Le modèle va donner une probabilité pour chaque label défini dans self.labels
            result = self.classifier(news, self.labels, multi_label=True)
            
            # Extraction des scores
            scores = dict(zip(result['labels'], result['scores']))
            
            # --- LOGIQUE CAUSALE (C'est ici qu'on fait de la finance) ---
            
            # 1. Score de "Supply Shock" (ex: Grève)
            supply_score = scores.get("supply disruption", 0.0)
            
            # 2. Score de "Geopolitics" (ex: Guerre)
            geo_score = scores.get("geopolitical tension", 0.0)
            
            # 3. Score de "Noise" (Bruit)
            noise_score = scores.get("market noise", 0.0)
            
            # Calcul du Signal Composite Z_t
            # Si c'est du bruit, on éteint le signal.
            # Sinon, on prend le max des risques.
            risk_intensity = max(supply_score, geo_score)
            
            if noise_score > risk_intensity:
                final_signal = 0.0
            else:
                # On amplifie le signal pour qu'il impacte bien le SDE (scale 0 à 5)
                final_signal = risk_intensity * 5.0
            
            signals.append(final_signal)
            
        return torch.tensor(signals).unsqueeze(-1) # [Batch, 1]