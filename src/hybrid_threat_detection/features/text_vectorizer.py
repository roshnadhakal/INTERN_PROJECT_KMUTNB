import os
import pickle
import logging
import ast
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from ..config.constants import RESULTS_DIR
from ..utils.helpers import create_dir_if_not_exists, save_json


logger = logging.getLogger(__name__)


class TextVectorizer:
    """Production-grade text vectorization with enhanced config handling."""
    
    def __init__(self, config: Dict):
        self.config = self._validate_config(config);
        self.vectorizers = {};
        self.tokenizers = {};
        self.embedding_matrix = None;
        self.vocab_size = None;
        self.max_len = None;
        
        # Create output directories
        self.vectorizer_dir = os.path.join(RESULTS_DIR, "phase2", "vectorizers");
        self.embedding_dir = os.path.join(RESULTS_DIR, "phase2", "embeddings");
        create_dir_if_not_exists(self.vectorizer_dir);
        create_dir_if_not_exists(self.embedding_dir);
        
        # Initialize NLTK
        self._init_nltk();


    def _init_nltk(self):
        """Initialize NLTK resources."""
        try:
            word_tokenize("test");
        except LookupError:
            import nltk;
            nltk.download('punkt');


    def _validate_config(self, config: Dict) -> Dict:
        """Validate and normalize configuration."""
        validated = config.copy();
        
        # Normalize TF-IDF config
        if 'tfidf_params' in validated:
            validated['tfidf_params'] = self._normalize_vectorizer_params(
                validated['tfidf_params']);
        
        # Normalize BOW config
        if 'bow_params' in validated:
            validated['bow_params'] = self._normalize_vectorizer_params(
                validated['bow_params']);
        
        # Verify dataset mappings
        if 'datasets' not in validated:
            raise ValueError("Configuration must contain 'datasets' mapping");
            
        return validated;


    def _normalize_vectorizer_params(self, params: Dict) -> Dict:
        """Normalize vectorizer parameters, especially ngram_range."""
        normalized = params.copy();
        
        if 'ngram_range' in normalized:
            normalized['ngram_range'] = self._parse_ngram_range(
                normalized['ngram_range']);
        
        return normalized;


    def _parse_ngram_range(self, ngram_spec: Any) -> Tuple[int, int]:
        """Convert various ngram_range specifications to tuple."""
        if isinstance(ngram_spec, tuple):
            return ngram_spec;
        elif isinstance(ngram_spec, list):
            return tuple(ngram_spec);
        elif isinstance(ngram_spec, str):
            try:
                parsed = ast.literal_eval(ngram_spec);
                if isinstance(parsed, (list, tuple)):
                    return tuple(parsed);
            except (ValueError, SyntaxError):
                pass;
        elif isinstance(ngram_spec, dict):
            return (ngram_spec.get('min', 1), ngram_spec.get('max', 2));
        
        raise ValueError(
            f"Cannot parse ngram_range specification: {ngram_spec}. "
            "Expected tuple, list, string representation, or dict with min/max");


    def vectorize(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Main vectorization method with enhanced error handling."""
        vectorized_data = {};
        
        for name, df in data.items():
            try:
                logger.info(f"Vectorizing dataset: {name}");
                
                # Get dataset-specific config
                dataset_config = self.config['datasets'].get(name, {});
                payload_col = dataset_config.get('payload_col', 'payload');
                
                if payload_col not in df.columns:
                    logger.warning(f"Payload column {payload_col} not found in {name}");
                    continue;
                
                texts = df[payload_col].astype(str).tolist();
                vec_results = {};
                
                # Traditional Vectorization
                if self.config.get('tfidf', False):
                    vec_results['tfidf'] = self._apply_tfidf(texts, name);
                
                if self.config.get('bow', False):
                    vec_results['bow'] = self._apply_bow(texts, name);
                
                # Deep Learning Vectorization
                dl_config = self.config.get('deep_learning', {});
                if dl_config.get('enabled', False):
                    dl_results = self._apply_dl_vectorization(texts, name, dl_config);
                    vec_results.update(dl_results);
                
                vectorized_data[name] = vec_results;
                
                # Save intermediate results
                self.save_vectorization_artifacts(name, vec_results);
                
            except Exception as e:
                logger.error(f"Error vectorizing {name}: {str(e)}", exc_info=True);
                raise;
        
        return vectorized_data;


    def _apply_tfidf(self, texts: list, dataset_name: str) -> Dict:
        """Apply TF-IDF vectorization with proper config handling."""
        logger.info(f"Applying TF-IDF to {dataset_name}");
        
        tfidf_config = self.config.get('tfidf_params', {
            'max_features': 5000,
            'ngram_range': (1, 2),
            'stop_words': 'english'
        });
        
        vectorizer = TfidfVectorizer(**tfidf_config);
        tfidf_matrix = vectorizer.fit_transform(texts);
        
        # Save vectorizer
        vec_path = os.path.join(self.vectorizer_dir, f"{dataset_name}_tfidf.pkl");
        with open(vec_path, 'wb') as f:
            pickle.dump(vectorizer, f);
        
        logger.info(f"TF-IDF completed for {dataset_name}. Shape: {tfidf_matrix.shape}");
        
        return {
            'matrix': tfidf_matrix,
            'vectorizer_path': vec_path,
            'feature_names': vectorizer.get_feature_names_out().tolist()
        };


    def _apply_bow(self, texts: list, dataset_name: str) -> Dict:
        """Apply Bag-of-Words vectorization."""
        logger.info(f"Applying BOW to {dataset_name}");
        
        bow_config = self.config.get('bow_params', {
            'max_features': 5000,
            'ngram_range': (1, 1),
            'stop_words': 'english'
        });
        
        vectorizer = CountVectorizer(**bow_config);
        bow_matrix = vectorizer.fit_transform(texts);
        
        # Save vectorizer
        vec_path = os.path.join(self.vectorizer_dir, f"{dataset_name}_bow.pkl");
        with open(vec_path, 'wb') as f:
            pickle.dump(vectorizer, f);
        
        logger.info(f"BOW completed for {dataset_name}. Shape: {bow_matrix.shape}");
        
        return {
            'matrix': bow_matrix,
            'vectorizer_path': vec_path,
            'feature_names': vectorizer.get_feature_names_out().tolist()
        };


    def _apply_dl_vectorization(self, texts: list, dataset_name: str, config: Dict) -> Dict:
        """Apply deep learning vectorization pipeline."""
        logger.info(f"Applying DL vectorization to {dataset_name}");
        
        results = {};
        
        # Tokenization
        tokenizer, seqs = self._tokenize_texts(texts, dataset_name, config);
        results['tokenizer'] = tokenizer;
        results['sequences'] = seqs;
        
        # Sequence padding
        padded_seqs = self._pad_sequences(seqs, config);
        results['padded_sequences'] = padded_seqs;
        
        # Embeddings
        if config.get('use_embeddings', False):
            embedding_matrix = self._load_embeddings(tokenizer, config);
            if embedding_matrix is not None:
                results['embedding_matrix'] = embedding_matrix;
                results['embedding_info'] = {
                    'vocab_size': self.vocab_size,
                    'embedding_dim': embedding_matrix.shape[1]
                };
        
        return results;


    def _tokenize_texts(self, texts: list, dataset_name: str, config: Dict) -> Tuple:
        """Tokenize texts and create sequences."""
        logger.info(f"Tokenizing texts for {dataset_name}");
        
        tokenizer_config = config.get('tokenizer_params', {
            'num_words': 10000,
            'oov_token': '<OOV>',
            'filters': '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        });
        
        tokenizer = Tokenizer(**tokenizer_config);
        tokenizer.fit_on_texts(texts);
        
        # Save tokenizer
        tokenizer_path = os.path.join(self.vectorizer_dir, f"{dataset_name}_tokenizer.pkl");
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f);
        
        # Convert texts to sequences
        sequences = tokenizer.texts_to_sequences(texts);
        
        # Update vocab size
        self.vocab_size = len(tokenizer.word_index) + 1;
        logger.info(f"Vocab size for {dataset_name}: {self.vocab_size}");
        
        return tokenizer, sequences;


    def _pad_sequences(self, sequences: list, config: Dict) -> np.ndarray:
        """Pad sequences to equal length."""
        max_len = config.get('max_len', 100);
        self.max_len = max_len;
        
        padded = pad_sequences(
            sequences,
            maxlen=max_len,
            padding='post',
            truncating='post'
        );
        
        logger.info(f"Padded sequences to length {max_len}. Shape: {padded.shape}");
        return padded;


    def _load_embeddings(self, tokenizer, config: Dict) -> Optional[np.ndarray]:
        """Load pretrained embeddings with enhanced error handling."""
        embedding_type = config.get('embedding_type', 'glove');
        embedding_dim = config.get('embedding_dim', 100);
        embedding_path = config.get('embedding_path');
        
        logger.info(f"Loading {embedding_type} embeddings with dim {embedding_dim}");
        
        try:
            if embedding_path and os.path.exists(embedding_path):
                embeddings_index = self._load_local_embeddings(embedding_path);
            else:
                embeddings_index = self._load_pretrained_embeddings(embedding_type, embedding_dim);
            
            # Create embedding matrix
            embedding_matrix = np.zeros((self.vocab_size, embedding_dim));
            
            for word, i in tokenizer.word_index.items():
                if i >= self.vocab_size:
                    continue;
                embedding_vector = embeddings_index.get(word);
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector;
            
            # Save embedding matrix
            matrix_path = os.path.join(self.embedding_dir, 'embedding_matrix.npy');
            np.save(matrix_path, embedding_matrix);
            
            logger.info(f"Created embedding matrix with shape {embedding_matrix.shape}");
            return embedding_matrix;
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {str(e)}");
            return None;
    def _load_embeddings(self, tokenizer, config: Dict) -> Optional[np.ndarray]:
        """Load pretrained embeddings and create embedding matrix."""
        embedding_type = config.get('embedding_type', 'glove')
        embedding_dim = config.get('embedding_dim', 100)
        embedding_path = config.get('embedding_path')
        
        logger.info(f"Loading {embedding_type} embeddings with dim {embedding_dim}")
        
        try:
            if embedding_path and os.path.exists(embedding_path):
                # Load from local file
                embeddings_index = self._load_local_embeddings(embedding_path)
            else:
                # Load from gensim's pretrained models
                embeddings_index = self._load_pretrained_embeddings(embedding_type, embedding_dim)
            
            # Create embedding matrix
            embedding_matrix = np.zeros((self.vocab_size, embedding_dim))
            
            for word, i in tokenizer.word_index.items():
                if i >= self.vocab_size:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            
            # Save embedding matrix
            matrix_path = os.path.join(self.embedding_dir, 'embedding_matrix.npy')
            np.save(matrix_path, embedding_matrix)
            
            logger.info(f"Created embedding matrix with shape {embedding_matrix.shape}")
            return embedding_matrix
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {str(e)}")
            return None
    
    def _load_local_embeddings(self, file_path: str) -> Dict:
        """Load embeddings from local text file."""
        embeddings_index = {}
        
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        logger.info(f"Loaded {len(embeddings_index)} word vectors from local file")
        return embeddings_index
    
    def _load_pretrained_embeddings(self, embedding_type: str, dim: int) -> Dict:
        """Load pretrained embeddings using gensim."""
        model_map = {
            'glove': f'glove-wiki-gigaword-{dim}',
            'word2vec': 'word2vec-google-news-300',
            'fasttext': 'fasttext-wiki-news-subwords-300'
        }
        
        model_name = model_map.get(embedding_type)
        if not model_name:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")
        
        logger.info(f"Downloading {model_name} embeddings...")
        embeddings = api.load(model_name)
        
        # Convert to dictionary format
        embeddings_index = {word: emb for word, emb in zip(embeddings.index_to_key, embeddings.vectors)}
        logger.info(f"Loaded {len(embeddings_index)} word vectors from {model_name}")
        
        return embeddings_index

    def save_vectorization_artifacts(self, dataset_name: str, results: Dict):
        """Save vectorization artifacts for a dataset."""
        artifacts_path = os.path.join(self.vectorizer_dir, f"{dataset_name}_artifacts.json");
        save_json({
            'vectorization_methods': list(results.keys()),
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'timestamp': pd.Timestamp.now().isoformat()
        }, artifacts_path);
    
