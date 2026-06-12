import torch
import torch.nn as nn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from typing import List, Tuple
import torch.nn.functional as F


class SemanticTextParser:
    """
    Tokenization-free semantic parser for non-segmented text (e.g., Burmese)
    using character-level n-grams and a Linear SVM.
    """
    def __init__(self, min_ngram: int = 2, max_ngram: int = 4):
        # Character-level n-grams handle lack of whitespace segmentation natively
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(min_ngram, max_ngram))
        # Calibrated via hinges/projections to output a proxy confidence score
        self.classifier = LinearSVC(C=1.0, dual=False, random_state=42)
        self.is_trained = False

    def fit(self, texts: List[str], labels: List[int]):
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True

    def parse(self, text: str) -> Tuple[torch.Tensor, float]:
        """
        Extracts both a dense continuous text feature vector 
        and a continuous semantic correctness scalar.
        """
        if not self.is_trained:
            raise ValueError("Parser must be trained before extracting features.")
            
        sparse_vec = self.vectorizer.transform([text])
        dense_feats = torch.tensor(sparse_vec.toarray(), dtype=torch.float32).squeeze(0)
        
        # Calculate decision function distance as a proxy for semantic correctness c_t
        decision_score = self.classifier.decision_function(sparse_vec)[0]
        # Squash into a continuous interval [0, 1]
        c_t = 1.0 / (1.0 + np.exp(-decision_score))
        
        return dense_feats, float(c_t)


class HybridNLPDKT(nn.Module):
    """
    Hybrid Deep Knowledge Tracing core fusing NLP feature embeddings 
    and multi-skill tracking via an LSTM loop.
    """
    def __init__(self, num_skills: int, text_dim: int, embed_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super(HybridNLPDKT, self).__init__()
        self.num_skills = num_skills
        
        # Latent categorical embedding for active skill tag
        self.skill_embeddings = nn.Embedding(num_skills, embed_dim)
        
        # Combined Input dimension: text features (v_text) + skill embedding + correctness scalar (c_t)
        # x_t = v_text || e(s_t) || c_t
        input_dim = text_dim + embed_dim + 1
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
        # Output layer simultaneously projecting hidden states to all concepts
        self.fully_connected = nn.Linear(hidden_dim, num_skills)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text_features: torch.Tensor, skill_indices: torch.Tensor, correctness_scores: torch.Tensor):
        """
        Args:
            text_features: Tensor of shape (batch_size, seq_len, text_dim)
            skill_indices: Tensor of shape (batch_size, seq_len)
            correctness_scores: Tensor of shape (batch_size, seq_len, 1)
        """
        batch_size, seq_len, _ = text_features.shape
        
        # 1. Project skills to low-dimensional space
        skill_embeds = self.skill_embeddings(skill_indices) # Shape: (batch_size, seq_len, embed_dim)
        
        # 2. Construct unified multimodal sequence tensor (x_t concatenation)
        x_t = torch.cat((text_features, skill_embeds, correctness_scores), dim=2)
        
        # 3. Recurrent trace propagation
        lstm_out, _ = self.lstm(x_t)
        lstm_out = self.dropout(lstm_out)
        
        # 4. Dense predictive mapping across the multi-skill matrix
        logits = self.fully_connected(lstm_out)
        return torch.sigmoid(logits)


# pipeline verification test
if __name__ == "__main__":
    print("Initializing components verification pipeline...")
    
    # Mock calibration data (Burmese algebra script representations)
    sample_corpus = [
        "x က တန်ဖိုး ၁၂ ဖြစ်ရမယ်", 
        "မသိကိန်းကို ပြောင်းပြန်လုပ်မယ်", 
        "အပေါင်းကို တစ်ဖက်ရွှေ့ရင် အနှုတ်ဖြစ်မယ်", 
        "အမှားဖြစ်သွားတယ် ၅ နဲ့ မြှောက်ရမှာ"
    ]
    sample_labels = [1, 1, 1, 0] # 1: Correct reasoning, 0: Misconception
    
    # 1. Test Parser Execution
    parser = SemanticTextParser(min_ngram=2, max_ngram=4)
    parser.fit(sample_corpus, sample_labels)
    v_text, c_t = parser.parse("x ကို ရှာဖို့ ၅ နဲ့ နှုတ်ရမယ်")
    
    text_features_dim = v_text.shape[0]
    print(f"Parser output dimensions successfully extracted: {text_features_dim}")
    print(f"Calculated target semantic correctness token scale: {c_t:.4f}")
    
    # 2. Test Model Evaluation Core
    NUM_SKILLS = 10
    EMBED_DIM = 32
    HIDDEN_DIM = 128
    
    model = HybridNLPDKT(num_skills=NUM_SKILLS, text_dim=text_features_dim, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
    
    # Synthesize small evaluation batches (batch_size=2, seq_len=3)
    mock_text_batch = torch.randn(2, 3, text_features_dim)
    mock_skill_batch = torch.randint(0, NUM_SKILLS, (2, 3))
    mock_ct_batch = torch.rand(2, 3, 1)
    
    predictions = model(mock_text_batch, mock_skill_batch, mock_ct_batch)
    print(f"Forward pass validated. Master profile shape: {predictions.shape} (Batch, Seq, Skills)")