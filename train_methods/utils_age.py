import csv 

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans as NPKMeans
from torch_kmeans import KMeans as TKMeans
from transformers import CLIPTokenizer, CLIPTextModel

from train_methods.consts import IMAGENET_1K, LEN_EN_3K_VOCAB, LEN_TOKENIZER_VOCAB

# TODO: 大部分はeapと同じなので共通化したい

def get_english_tokens():
    data_path = 'data/english_3000.csv'
    df = pd.read_csv(data_path)
    vocab = {}
    for ir, row in df.iterrows():
        vocab[row['word']] = ir
    assert(len(vocab) == LEN_EN_3K_VOCAB)
    return vocab

def get_imagenet_tokens():
    # 1: 'goldfish, Carassius auratus',
    # 2: 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
    vocab = {}
    for index in IMAGENET_1K:
        token = IMAGENET_1K[index] 
        vocab[token] = index
    return vocab

@torch.no_grad()
def get_vocab(tokenizer: CLIPTokenizer, model_name, vocab='EN3K'):
    if vocab == 'CLIP':
        if model_name == 'SD-v1-4':
            return tokenizer.get_vocab()
        elif model_name == 'SD-v2-1':
            return tokenizer.encoder
        else:
            raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
    elif vocab == 'EN3K':
        return get_english_tokens()
    elif vocab == 'Imagenet':
        return get_imagenet_tokens()
    else:
        raise ValueError("vocab should be either 'CLIP' or 'EN3K' or 'Imagenet'")

@torch.no_grad()
def get_condition(prompt: str, tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel) -> torch.Tensor:
    token_ids = tokenizer.encode(
        [prompt], 
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids
    return text_encoder(token_ids.to(text_encoder.device))[0]

@torch.no_grad()
def create_embedding_matrix(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    start=0,
    end=LEN_TOKENIZER_VOCAB,
    model_name='SD-v1-4',
    save_mode='array',
    remove_end_token=False,
    vocab='EN3K'
):

    if type(vocab) == str:
        tokenizer_vocab = get_vocab(tokenizer, model_name, vocab=vocab)
    else:
        tokenizer_vocab = vocab

    print(f"tokenizer_vocab: {tokenizer_vocab}")

    if save_mode == 'array':
        all_embeddings = []
        for token in tokenizer_vocab:
            if tokenizer_vocab[token] < start or tokenizer_vocab[token] >= end:
                continue
            if remove_end_token:
                token_ = token.replace('</w>','')
            else:
                token_ = token
            emb_ = get_condition(token_, tokenizer, text_encoder)
            all_embeddings.append(emb_)
        return torch.cat(all_embeddings, dim=0) # shape (49408, 77, 768)
    elif save_mode == 'dict':
        all_embeddings = {}
        for token in tokenizer_vocab:
            if tokenizer_vocab[token] < start or tokenizer_vocab[token] >= end:
                continue
            if remove_end_token:
                token_ = token.replace('</w>','')
            else:
                token_ = token
            emb_ = get_condition(token_, tokenizer, text_encoder)
            all_embeddings[token] = emb_
        return all_embeddings
    else:
        raise ValueError("save_mode should be either 'array' or 'dict'")

def get_similarities(sim: str, concept_embedding: torch.Tensor, embedding_matrix: torch.Tensor) -> torch.Tensor:
    if sim == 'cosine':
        return F.cosine_similarity(concept_embedding, embedding_matrix, dim=-1)
    elif sim == 'l2':
        return - F.pairwise_distance(concept_embedding, embedding_matrix, p=2)

def detect_special_tokens(text: str) -> bool:
    text = text.lower()
    for i in range(len(text)):
        if text[i] not in 'abcdefghijklmnopqrstuvwxyz</> ': # include space
            return True
    return False

@torch.no_grad()
def search_closest_tokens(
    concept: str, 
    tokenizer: CLIPTokenizer, 
    text_encoder: CLIPTextModel, 
    k: int=5, 
    reshape: bool=True, 
    sim: str='cosine', 
    model_name: str='SD-v1-4', 
    ignore_special_tokens: bool=True, 
    vocab: str='EN3K'
):
    """
    Given a concept, i.e., "nudity", search for top-k closest tokens in the embedding space
    """
    if type(vocab) == str:
        tokenizer_vocab = get_vocab(tokenizer, model_name, vocab=vocab)
    else:
        tokenizer_vocab = vocab
    # inverse the dictionary
    tokenizer_vocab_indexing = {v: k for k, v in tokenizer_vocab.items()}

    concept_embedding: torch.Tensor = get_condition(concept, tokenizer, text_encoder)

    # Calculate the cosine similarity between the concept and all tokens
    # load the embedding matrix 
    all_similarities = []
    
    if vocab == 'CLIP':
        for start in range(0, LEN_TOKENIZER_VOCAB, 5000):
            end = min(LEN_TOKENIZER_VOCAB, start+5000)
            if model_name == 'SD-v1-4':
                embedding_matrix: torch.Tensor = torch.load(f'models/embedding_matrix_{start}_{end}_array.pt')
            elif model_name == 'SD-v2-1':
                embedding_matrix: torch.Tensor = torch.load(f'models/embedding_matrix_{start}_{end}_array_v2-1.pt')
            else:
                raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
            
            if reshape:
                concept_embedding = concept_embedding.view(concept_embedding.size(0), -1)
                embedding_matrix = embedding_matrix.view(embedding_matrix.size(0), -1)
            similarities = get_similarities(sim, concept_embedding, embedding_matrix)
            all_similarities.append(similarities)
    elif vocab == 'EN3K':
        if model_name == 'SD-v1-4':
            embedding_matrix = torch.load(f'models/embedding_matrix_array_EN3K.pt')
        elif model_name == 'SD-v2-1':
            embedding_matrix = torch.load(f'models/embedding_matrix_array_EN3K_v2-1.pt')
        else:
            raise ValueError("model_name should be either 'SD-v1-4' or 'SD-v2-1'")
        if reshape:
            concept_embedding = concept_embedding.view(concept_embedding.size(0), -1)
            embedding_matrix = embedding_matrix.view(embedding_matrix.size(0), -1)
        similarities = get_similarities(sim, concept_embedding, embedding_matrix)
        all_similarities.append(similarities)
    
    elif vocab == 'Imagenet':
        embedding_matrix = create_embedding_matrix(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            start=0,
            end=1000,
            model_name=model_name,
            save_mode='array',
            vocab='Imagenet'
        )
        if reshape:
            concept_embedding = concept_embedding.view(concept_embedding.size(0), -1)
            embedding_matrix = embedding_matrix.view(embedding_matrix.size(0), -1)
        similarities = get_similarities(sim, concept_embedding, embedding_matrix)
        all_similarities.append(similarities)
    
    elif type(vocab) == dict:
        embedding_matrix = create_embedding_matrix(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            start=0,
            end=len(vocab),
            model_name=model_name,
            save_mode='array',
            vocab=vocab
        )
        if reshape:
            concept_embedding = concept_embedding.view(concept_embedding.size(0), -1)
            embedding_matrix = embedding_matrix.view(embedding_matrix.size(0), -1)
        similarities = get_similarities(sim, concept_embedding, embedding_matrix)
        all_similarities.append(similarities)
    
    else:
        raise ValueError("vocab should be either 'CLIP' or 'EN3K' or 'Imagenet' or a dictionary")

    similarities = torch.cat(all_similarities, dim=0)
    # sorting the similarities
    sorted_similarities, indices = torch.sort(similarities, descending=True)
    print(f"{sorted_similarities[:10]=}")
    print(f"{indices[:10]=}")
    print(f"{tokenizer_vocab_indexing=}")

    sim_dict = {}
    for im, i in enumerate(indices):
        if i.item() not in tokenizer_vocab_indexing:
            print('not in tokenizer_vocab_indexing: ', i.item())
            continue
        if ignore_special_tokens:
            if detect_special_tokens(tokenizer_vocab_indexing[i.item()]):
                print('detect_special_tokens: ', tokenizer_vocab_indexing[i.item()])
                continue
        token = tokenizer_vocab_indexing[i.item()]
        sim_dict[token] = sorted_similarities[im]
    
    print('sim_dict: ', sim_dict)
    top_k_tokens = list(sim_dict.keys())[:k]
    print(f"Top-{k} closest tokens to the concept {concept} are: {top_k_tokens}")
    return top_k_tokens, sim_dict

@torch.no_grad()
def save_embedding_matrix(tokenizer: CLIPTokenizer, text_encoder: CLIPTextModel, model_name='SD-v1-4', save_mode='array', vocab='EN3K'):
    if vocab == 'CLIP':
        for start in range(0, LEN_TOKENIZER_VOCAB, 5000):
            print(f"start: {start} / {LEN_TOKENIZER_VOCAB}")
            end = min(LEN_TOKENIZER_VOCAB, start+5000)
            embedding_matrix = create_embedding_matrix(tokenizer, text_encoder, start=start, end=end, model_name=model_name, save_mode=save_mode)
            if model_name == 'SD-v1-4':
                torch.save(embedding_matrix, f'../Adversarial_Erasure/models/embedding_matrix_{start}_{end}_{save_mode}.pt')
            elif model_name == 'SD-v2-1':
                torch.save(embedding_matrix, f'../Adversarial_Erasure/models/embedding_matrix_{start}_{end}_{save_mode}_v2-1.pt')
    
    elif vocab == 'EN3K':
        embedding_matrix = create_embedding_matrix(tokenizer, text_encoder, start=0, end=LEN_EN_3K_VOCAB, model_name=model_name, save_mode=save_mode, vocab='EN3K')
        if model_name == 'SD-v1-4':
            torch.save(embedding_matrix, f'models/embedding_matrix_{save_mode}_EN3K.pt')
        elif model_name == 'SD-v2-1':
            torch.save(embedding_matrix, f'models/embedding_matrix_{save_mode}_EN3K_v2-1.pt')
    
    elif vocab == 'Imagenet':
        embedding_matrix = create_embedding_matrix(tokenizer, text_encoder, start=0, end=1000, model_name=model_name, save_mode=save_mode, vocab='Imagenet')
        if model_name == 'SD-v1-4':
            torch.save(embedding_matrix, f'models/embedding_matrix_{save_mode}_Imagenet.pt')
        elif model_name == 'SD-v2-1':
            torch.save(embedding_matrix, f'models/embedding_matrix_{save_mode}_Imagenet_v2-1.pt')

    else:
        raise ValueError("vocab should be either 'CLIP' or 'EN3K'")

def my_kmean(sorted_sim_dict, num_centers, compute_mode):
    if compute_mode == 'numpy':
        similarities = np.array([sorted_sim_dict[token].item() for token in sorted_sim_dict])
        similarities = similarities.reshape(-1, 1)
        kmeans = NPKMeans(n_clusters=num_centers, random_state=0).fit(similarities)
        print(f"{kmeans.cluster_centers_=}")
        print(f"{kmeans.labels_=}")
        cluster_centers = kmeans.cluster_centers_
    elif compute_mode == 'torch':
        similarities = torch.stack([sorted_sim_dict[token] for token in sorted_sim_dict])
        similarities = torch.unsqueeze(similarities, dim=0)
        similarities = torch.unsqueeze(similarities, dim=2) # [1, N, 1]
        print('similarities shape:', similarities.shape)
        kmeans = TKMeans(n_clusters=num_centers).fit(similarities)
        print(f"{kmeans.cluster_centers=}")
        print(f"{kmeans.labels=}")
        cluster_centers = kmeans.cluster_centers

    # find the closest token to each cluster center
    cluster_dict = {}
    for i, center in enumerate(cluster_centers):
        closest_token = None
        closest_similarity = -float('inf')
        for j, token in enumerate(sorted_sim_dict):
            similarity = sorted_sim_dict[token].item()
            if abs(similarity - center) < abs(closest_similarity - center):
                closest_similarity = similarity
                closest_token = token
        cluster_dict[closest_token] = (closest_token, closest_similarity, i)
    print(f"{cluster_dict=}")

    return cluster_dict

@torch.no_grad()
def learn_k_means_from_input_embedding(sim_dict: dict, num_centers=5, compute_mode='numpy'):
    """
    Given a model, a set of tokens, and a concept, learn k-means clustering on the search_closest_tokens's output
    """
    if num_centers <= 0:
        print(f"Number of centers should be greater than 0. Returning the tokens themselves.")
        return list(sim_dict.keys())
    if len(list(sim_dict.keys())) <= num_centers:
        print(f"Number of tokens is less than the number of centers. Returning the tokens themselves.")
        return list(sim_dict.keys())

    return list(my_kmean(sim_dict, num_centers, compute_mode).keys())

class ConceptDict:
    def __init__(self):
        self.all_concepts = {}

    def load_concepts(self, concept_name, csv_file_path):
        
        data = []
        with open(csv_file_path, mode='r') as file:
            reader = csv.reader(file)
            # next(reader)  # Skip the header
            for row in reader:
                data.append(row[0])

        if concept_name not in self.all_concepts:
            self.all_concepts[concept_name] = []

        self.all_concepts[concept_name].extend(data)

    def load_imagenet_concepts(self):
        # 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias'
        concepts = []
        for index in IMAGENET_1K:
            concept = IMAGENET_1K[index]
            concepts.append(concept.split(',')[0])
        
        self.all_concepts['imagenet'] = concepts

    def get_concepts(self, concept_name):
        if concept_name not in self.all_concepts:
            raise ValueError(f"Concept name '{concept_name}' not found in the dictionary.") 
        return self.all_concepts[concept_name]

    def get_concepts_as_dict(self, concept_name):
        # format: vocab[token] = index
        if concept_name not in self.all_concepts:
            raise ValueError(f"Concept name '{concept_name}' not found in the dictionary.") 
        vocab = {}
        for index, token in enumerate(self.all_concepts[concept_name]):
            vocab[token] = index
        return vocab

    def load_all_concepts(self):
        self.load_concepts('Cassette Player', 'captions/imagenette_cassette_player.csv')
        self.load_concepts('Chain Saw', 'captions/imagenette_chain_saw.csv')
        self.load_concepts('Church', 'captions/imagenette_church.csv')
        self.load_concepts('Gas Pump', 'captions/imagenette_gas_pump.csv')
        self.load_concepts('Tench', 'captions/imagenette_tench.csv')
        self.load_concepts('Garbage Truck', 'captions/imagenette_garbage_truck.csv')
        self.load_concepts('English Springer', 'captions/imagenette_english_springer.csv')
        self.load_concepts('Golf Ball', 'captions/imagenette_golf_ball.csv')
        self.load_concepts('parachute', 'captions/imagenette_parachute.csv')
        self.load_concepts('French Horn', 'captions/imagenette_french_horn.csv')

        self.load_concepts('imagenette', 'captions/imagenette_cassette_player.csv')
        self.load_concepts('imagenette', 'captions/imagenette_chain_saw.csv')
        self.load_concepts('imagenette', 'captions/imagenette_church.csv')
        self.load_concepts('imagenette', 'captions/imagenette_gas_pump.csv')
        self.load_concepts('imagenette', 'captions/imagenette_tench.csv')
        self.load_concepts('imagenette', 'captions/imagenette_garbage_truck.csv')
        self.load_concepts('imagenette', 'captions/imagenette_english_springer.csv')
        self.load_concepts('imagenette', 'captions/imagenette_golf_ball.csv')
        self.load_concepts('imagenette', 'captions/imagenette_parachute.csv')
        self.load_concepts('imagenette', 'captions/imagenette_french_horn.csv')

        self.load_concepts('nudity', 'captions/nudity_human_body.csv')
        self.load_concepts('nudity', 'captions/nudity_naked_person.csv')
        self.load_concepts('nudity', 'captions/nudity_nude_person.csv')

        self.load_concepts('human_body', 'captions/nudity_human_body.csv')

        self.load_concepts('artistic', 'captions/artistic_concepts.csv')
        self.load_concepts('artistic', 'captions/artistic_painting.csv')

        self.load_imagenet_concepts()

