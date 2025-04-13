import csv 

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans as NPKMeans
from torch_kmeans import KMeans as TKMeans

from transformers import CLIPTokenizer, CLIPTextModel

# constants
LEN_EN_3K_VOCAB = 3000
LEN_TOKENIZER_VOCAB = 49408

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

IMAGENET_1K = {0: 'tench, Tinca tinca',
 1: 'goldfish, Carassius auratus',
 2: 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
 3: 'tiger shark, Galeocerdo cuvieri',
 4: 'hammerhead, hammerhead shark',
 5: 'electric ray, crampfish, numbfish, torpedo',
 6: 'stingray',
 7: 'cock',
 8: 'hen',
 9: 'ostrich, Struthio camelus',
 10: 'brambling, Fringilla montifringilla',
 11: 'goldfinch, Carduelis carduelis',
 12: 'house finch, linnet, Carpodacus mexicanus',
 13: 'junco, snowbird',
 14: 'indigo bunting, indigo finch, indigo bird, Passerina cyanea',
 15: 'robin, American robin, Turdus migratorius',
 16: 'bulbul',
 17: 'jay',
 18: 'magpie',
 19: 'chickadee',
 20: 'water ouzel, dipper',
 21: 'kite',
 22: 'bald eagle, American eagle, Haliaeetus leucocephalus',
 23: 'vulture',
 24: 'great grey owl, great gray owl, Strix nebulosa',
 25: 'European fire salamander, Salamandra salamandra',
 26: 'common newt, Triturus vulgaris',
 27: 'eft',
 28: 'spotted salamander, Ambystoma maculatum',
 29: 'axolotl, mud puppy, Ambystoma mexicanum',
 30: 'bullfrog, Rana catesbeiana',
 31: 'tree frog, tree-frog',
 32: 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
 33: 'loggerhead, loggerhead turtle, Caretta caretta',
 34: 'leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea',
 35: 'mud turtle',
 36: 'terrapin',
 37: 'box turtle, box tortoise',
 38: 'banded gecko',
 39: 'common iguana, iguana, Iguana iguana',
 40: 'American chameleon, anole, Anolis carolinensis',
 41: 'whiptail, whiptail lizard',
 42: 'agama',
 43: 'frilled lizard, Chlamydosaurus kingi',
 44: 'alligator lizard',
 45: 'Gila monster, Heloderma suspectum',
 46: 'green lizard, Lacerta viridis',
 47: 'African chameleon, Chamaeleo chamaeleon',
 48: 'Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis',
 49: 'African crocodile, Nile crocodile, Crocodylus niloticus',
 50: 'American alligator, Alligator mississipiensis',
 51: 'triceratops',
 52: 'thunder snake, worm snake, Carphophis amoenus',
 53: 'ringneck snake, ring-necked snake, ring snake',
 54: 'hognose snake, puff adder, sand viper',
 55: 'green snake, grass snake',
 56: 'king snake, kingsnake',
 57: 'garter snake, grass snake',
 58: 'water snake',
 59: 'vine snake',
 60: 'night snake, Hypsiglena torquata',
 61: 'boa constrictor, Constrictor constrictor',
 62: 'rock python, rock snake, Python sebae',
 63: 'Indian cobra, Naja naja',
 64: 'green mamba',
 65: 'sea snake',
 66: 'horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
 67: 'diamondback, diamondback rattlesnake, Crotalus adamanteus',
 68: 'sidewinder, horned rattlesnake, Crotalus cerastes',
 69: 'trilobite',
 70: 'harvestman, daddy longlegs, Phalangium opilio',
 71: 'scorpion',
 72: 'black and gold garden spider, Argiope aurantia',
 73: 'barn spider, Araneus cavaticus',
 74: 'garden spider, Aranea diademata',
 75: 'black widow, Latrodectus mactans',
 76: 'tarantula',
 77: 'wolf spider, hunting spider',
 78: 'tick',
 79: 'centipede',
 80: 'black grouse',
 81: 'ptarmigan',
 82: 'ruffed grouse, partridge, Bonasa umbellus',
 83: 'prairie chicken, prairie grouse, prairie fowl',
 84: 'peacock',
 85: 'quail',
 86: 'partridge',
 87: 'African grey, African gray, Psittacus erithacus',
 88: 'macaw',
 89: 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita',
 90: 'lorikeet',
 91: 'coucal',
 92: 'bee eater',
 93: 'hornbill',
 94: 'hummingbird',
 95: 'jacamar',
 96: 'toucan',
 97: 'drake',
 98: 'red-breasted merganser, Mergus serrator',
 99: 'goose',
 100: 'black swan, Cygnus atratus',
 101: 'tusker',
 102: 'echidna, spiny anteater, anteater',
 103: 'platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus',
 104: 'wallaby, brush kangaroo',
 105: 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
 106: 'wombat',
 107: 'jellyfish',
 108: 'sea anemone, anemone',
 109: 'brain coral',
 110: 'flatworm, platyhelminth',
 111: 'nematode, nematode worm, roundworm',
 112: 'conch',
 113: 'snail',
 114: 'slug',
 115: 'sea slug, nudibranch',
 116: 'chiton, coat-of-mail shell, sea cradle, polyplacophore',
 117: 'chambered nautilus, pearly nautilus, nautilus',
 118: 'Dungeness crab, Cancer magister',
 119: 'rock crab, Cancer irroratus',
 120: 'fiddler crab',
 121: 'king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica',
 122: 'American lobster, Northern lobster, Maine lobster, Homarus americanus',
 123: 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
 124: 'crayfish, crawfish, crawdad, crawdaddy',
 125: 'hermit crab',
 126: 'isopod',
 127: 'white stork, Ciconia ciconia',
 128: 'black stork, Ciconia nigra',
 129: 'spoonbill',
 130: 'flamingo',
 131: 'little blue heron, Egretta caerulea',
 132: 'American egret, great white heron, Egretta albus',
 133: 'bittern',
 134: 'crane',
 135: 'limpkin, Aramus pictus',
 136: 'European gallinule, Porphyrio porphyrio',
 137: 'American coot, marsh hen, mud hen, water hen, Fulica americana',
 138: 'bustard',
 139: 'ruddy turnstone, Arenaria interpres',
 140: 'red-backed sandpiper, dunlin, Erolia alpina',
 141: 'redshank, Tringa totanus',
 142: 'dowitcher',
 143: 'oystercatcher, oyster catcher',
 144: 'pelican',
 145: 'king penguin, Aptenodytes patagonica',
 146: 'albatross, mollymawk',
 147: 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus',
 148: 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca',
 149: 'dugong, Dugong dugon',
 150: 'sea lion',
 151: 'Chihuahua',
 152: 'Japanese spaniel',
 153: 'Maltese dog, Maltese terrier, Maltese',
 154: 'Pekinese, Pekingese, Peke',
 155: 'Shih-Tzu',
 156: 'Blenheim spaniel',
 157: 'papillon',
 158: 'toy terrier',
 159: 'Rhodesian ridgeback',
 160: 'Afghan hound, Afghan',
 161: 'basset, basset hound',
 162: 'beagle',
 163: 'bloodhound, sleuthhound',
 164: 'bluetick',
 165: 'black-and-tan coonhound',
 166: 'Walker hound, Walker foxhound',
 167: 'English foxhound',
 168: 'redbone',
 169: 'borzoi, Russian wolfhound',
 170: 'Irish wolfhound',
 171: 'Italian greyhound',
 172: 'whippet',
 173: 'Ibizan hound, Ibizan Podenco',
 174: 'Norwegian elkhound, elkhound',
 175: 'otterhound, otter hound',
 176: 'Saluki, gazelle hound',
 177: 'Scottish deerhound, deerhound',
 178: 'Weimaraner',
 179: 'Staffordshire bullterrier, Staffordshire bull terrier',
 180: 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier',
 181: 'Bedlington terrier',
 182: 'Border terrier',
 183: 'Kerry blue terrier',
 184: 'Irish terrier',
 185: 'Norfolk terrier',
 186: 'Norwich terrier',
 187: 'Yorkshire terrier',
 188: 'wire-haired fox terrier',
 189: 'Lakeland terrier',
 190: 'Sealyham terrier, Sealyham',
 191: 'Airedale, Airedale terrier',
 192: 'cairn, cairn terrier',
 193: 'Australian terrier',
 194: 'Dandie Dinmont, Dandie Dinmont terrier',
 195: 'Boston bull, Boston terrier',
 196: 'miniature schnauzer',
 197: 'giant schnauzer',
 198: 'standard schnauzer',
 199: 'Scotch terrier, Scottish terrier, Scottie',
 200: 'Tibetan terrier, chrysanthemum dog',
 201: 'silky terrier, Sydney silky',
 202: 'soft-coated wheaten terrier',
 203: 'West Highland white terrier',
 204: 'Lhasa, Lhasa apso',
 205: 'flat-coated retriever',
 206: 'curly-coated retriever',
 207: 'golden retriever',
 208: 'Labrador retriever',
 209: 'Chesapeake Bay retriever',
 210: 'German short-haired pointer',
 211: 'vizsla, Hungarian pointer',
 212: 'English setter',
 213: 'Irish setter, red setter',
 214: 'Gordon setter',
 215: 'Brittany spaniel',
 216: 'clumber, clumber spaniel',
 217: 'English springer, English springer spaniel',
 218: 'Welsh springer spaniel',
 219: 'cocker spaniel, English cocker spaniel, cocker',
 220: 'Sussex spaniel',
 221: 'Irish water spaniel',
 222: 'kuvasz',
 223: 'schipperke',
 224: 'groenendael',
 225: 'malinois',
 226: 'briard',
 227: 'kelpie',
 228: 'komondor',
 229: 'Old English sheepdog, bobtail',
 230: 'Shetland sheepdog, Shetland sheep dog, Shetland',
 231: 'collie',
 232: 'Border collie',
 233: 'Bouvier des Flandres, Bouviers des Flandres',
 234: 'Rottweiler',
 235: 'German shepherd, German shepherd dog, German police dog, alsatian',
 236: 'Doberman, Doberman pinscher',
 237: 'miniature pinscher',
 238: 'Greater Swiss Mountain dog',
 239: 'Bernese mountain dog',
 240: 'Appenzeller',
 241: 'EntleBucher',
 242: 'boxer',
 243: 'bull mastiff',
 244: 'Tibetan mastiff',
 245: 'French bulldog',
 246: 'Great Dane',
 247: 'Saint Bernard, St Bernard',
 248: 'Eskimo dog, husky',
 249: 'malamute, malemute, Alaskan malamute',
 250: 'Siberian husky',
 251: 'dalmatian, coach dog, carriage dog',
 252: 'affenpinscher, monkey pinscher, monkey dog',
 253: 'basenji',
 254: 'pug, pug-dog',
 255: 'Leonberg',
 256: 'Newfoundland, Newfoundland dog',
 257: 'Great Pyrenees',
 258: 'Samoyed, Samoyede',
 259: 'Pomeranian',
 260: 'chow, chow chow',
 261: 'keeshond',
 262: 'Brabancon griffon',
 263: 'Pembroke, Pembroke Welsh corgi',
 264: 'Cardigan, Cardigan Welsh corgi',
 265: 'toy poodle',
 266: 'miniature poodle',
 267: 'standard poodle',
 268: 'Mexican hairless',
 269: 'timber wolf, grey wolf, gray wolf, Canis lupus',
 270: 'white wolf, Arctic wolf, Canis lupus tundrarum',
 271: 'red wolf, maned wolf, Canis rufus, Canis niger',
 272: 'coyote, prairie wolf, brush wolf, Canis latrans',
 273: 'dingo, warrigal, warragal, Canis dingo',
 274: 'dhole, Cuon alpinus',
 275: 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus',
 276: 'hyena, hyaena',
 277: 'red fox, Vulpes vulpes',
 278: 'kit fox, Vulpes macrotis',
 279: 'Arctic fox, white fox, Alopex lagopus',
 280: 'grey fox, gray fox, Urocyon cinereoargenteus',
 281: 'tabby, tabby cat',
 282: 'tiger cat',
 283: 'Persian cat',
 284: 'Siamese cat, Siamese',
 285: 'Egyptian cat',
 286: 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
 287: 'lynx, catamount',
 288: 'leopard, Panthera pardus',
 289: 'snow leopard, ounce, Panthera uncia',
 290: 'jaguar, panther, Panthera onca, Felis onca',
 291: 'lion, king of beasts, Panthera leo',
 292: 'tiger, Panthera tigris',
 293: 'cheetah, chetah, Acinonyx jubatus',
 294: 'brown bear, bruin, Ursus arctos',
 295: 'American black bear, black bear, Ursus americanus, Euarctos americanus',
 296: 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus',
 297: 'sloth bear, Melursus ursinus, Ursus ursinus',
 298: 'mongoose',
 299: 'meerkat, mierkat',
 300: 'tiger beetle',
 301: 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
 302: 'ground beetle, carabid beetle',
 303: 'long-horned beetle, longicorn, longicorn beetle',
 304: 'leaf beetle, chrysomelid',
 305: 'dung beetle',
 306: 'rhinoceros beetle',
 307: 'weevil',
 308: 'fly',
 309: 'bee',
 310: 'ant, emmet, pismire',
 311: 'grasshopper, hopper',
 312: 'cricket',
 313: 'walking stick, walkingstick, stick insect',
 314: 'cockroach, roach',
 315: 'mantis, mantid',
 316: 'cicada, cicala',
 317: 'leafhopper',
 318: 'lacewing, lacewing fly',
 319: "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
 320: 'damselfly',
 321: 'admiral',
 322: 'ringlet, ringlet butterfly',
 323: 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
 324: 'cabbage butterfly',
 325: 'sulphur butterfly, sulfur butterfly',
 326: 'lycaenid, lycaenid butterfly',
 327: 'starfish, sea star',
 328: 'sea urchin',
 329: 'sea cucumber, holothurian',
 330: 'wood rabbit, cottontail, cottontail rabbit',
 331: 'hare',
 332: 'Angora, Angora rabbit',
 333: 'hamster',
 334: 'porcupine, hedgehog',
 335: 'fox squirrel, eastern fox squirrel, Sciurus niger',
 336: 'marmot',
 337: 'beaver',
 338: 'guinea pig, Cavia cobaya',
 339: 'sorrel',
 340: 'zebra',
 341: 'hog, pig, grunter, squealer, Sus scrofa',
 342: 'wild boar, boar, Sus scrofa',
 343: 'warthog',
 344: 'hippopotamus, hippo, river horse, Hippopotamus amphibius',
 345: 'ox',
 346: 'water buffalo, water ox, Asiatic buffalo, Bubalus bubalis',
 347: 'bison',
 348: 'ram, tup',
 349: 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
 350: 'ibex, Capra ibex',
 351: 'hartebeest',
 352: 'impala, Aepyceros melampus',
 353: 'gazelle',
 354: 'Arabian camel, dromedary, Camelus dromedarius',
 355: 'llama',
 356: 'weasel',
 357: 'mink',
 358: 'polecat, fitch, foulmart, foumart, Mustela putorius',
 359: 'black-footed ferret, ferret, Mustela nigripes',
 360: 'otter',
 361: 'skunk, polecat, wood pussy',
 362: 'badger',
 363: 'armadillo',
 364: 'three-toed sloth, ai, Bradypus tridactylus',
 365: 'orangutan, orang, orangutang, Pongo pygmaeus',
 366: 'gorilla, Gorilla gorilla',
 367: 'chimpanzee, chimp, Pan troglodytes',
 368: 'gibbon, Hylobates lar',
 369: 'siamang, Hylobates syndactylus, Symphalangus syndactylus',
 370: 'guenon, guenon monkey',
 371: 'patas, hussar monkey, Erythrocebus patas',
 372: 'baboon',
 373: 'macaque',
 374: 'langur',
 375: 'colobus, colobus monkey',
 376: 'proboscis monkey, Nasalis larvatus',
 377: 'marmoset',
 378: 'capuchin, ringtail, Cebus capucinus',
 379: 'howler monkey, howler',
 380: 'titi, titi monkey',
 381: 'spider monkey, Ateles geoffroyi',
 382: 'squirrel monkey, Saimiri sciureus',
 383: 'Madagascar cat, ring-tailed lemur, Lemur catta',
 384: 'indri, indris, Indri indri, Indri brevicaudatus',
 385: 'Indian elephant, Elephas maximus',
 386: 'African elephant, Loxodonta africana',
 387: 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
 388: 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca',
 389: 'barracouta, snoek',
 390: 'eel',
 391: 'coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch',
 392: 'rock beauty, Holocanthus tricolor',
 393: 'anemone fish',
 394: 'sturgeon',
 395: 'gar, garfish, garpike, billfish, Lepisosteus osseus',
 396: 'lionfish',
 397: 'puffer, pufferfish, blowfish, globefish',
 398: 'abacus',
 399: 'abaya',
 400: "academic gown, academic robe, judge's robe",
 401: 'accordion, piano accordion, squeeze box',
 402: 'acoustic guitar',
 403: 'aircraft carrier, carrier, flattop, attack aircraft carrier',
 404: 'airliner',
 405: 'airship, dirigible',
 406: 'altar',
 407: 'ambulance',
 408: 'amphibian, amphibious vehicle',
 409: 'analog clock',
 410: 'apiary, bee house',
 411: 'apron',
 412: 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin',
 413: 'assault rifle, assault gun',
 414: 'backpack, back pack, knapsack, packsack, rucksack, haversack',
 415: 'bakery, bakeshop, bakehouse',
 416: 'balance beam, beam',
 417: 'balloon',
 418: 'ballpoint, ballpoint pen, ballpen, Biro',
 419: 'Band Aid',
 420: 'banjo',
 421: 'bannister, banister, balustrade, balusters, handrail',
 422: 'barbell',
 423: 'barber chair',
 424: 'barbershop',
 425: 'barn',
 426: 'barometer',
 427: 'barrel, cask',
 428: 'barrow, garden cart, lawn cart, wheelbarrow',
 429: 'baseball',
 430: 'basketball',
 431: 'bassinet',
 432: 'bassoon',
 433: 'bathing cap, swimming cap',
 434: 'bath towel',
 435: 'bathtub, bathing tub, bath, tub',
 436: 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
 437: 'beacon, lighthouse, beacon light, pharos',
 438: 'beaker',
 439: 'bearskin, busby, shako',
 440: 'beer bottle',
 441: 'beer glass',
 442: 'bell cote, bell cot',
 443: 'bib',
 444: 'bicycle-built-for-two, tandem bicycle, tandem',
 445: 'bikini, two-piece',
 446: 'binder, ring-binder',
 447: 'binoculars, field glasses, opera glasses',
 448: 'birdhouse',
 449: 'boathouse',
 450: 'bobsled, bobsleigh, bob',
 451: 'bolo tie, bolo, bola tie, bola',
 452: 'bonnet, poke bonnet',
 453: 'bookcase',
 454: 'bookshop, bookstore, bookstall',
 455: 'bottlecap',
 456: 'bow',
 457: 'bow tie, bow-tie, bowtie',
 458: 'brass, memorial tablet, plaque',
 459: 'brassiere, bra, bandeau',
 460: 'breakwater, groin, groyne, mole, bulwark, seawall, jetty',
 461: 'breastplate, aegis, egis',
 462: 'broom',
 463: 'bucket, pail',
 464: 'buckle',
 465: 'bulletproof vest',
 466: 'bullet train, bullet',
 467: 'butcher shop, meat market',
 468: 'cab, hack, taxi, taxicab',
 469: 'caldron, cauldron',
 470: 'candle, taper, wax light',
 471: 'cannon',
 472: 'canoe',
 473: 'can opener, tin opener',
 474: 'cardigan',
 475: 'car mirror',
 476: 'carousel, carrousel, merry-go-round, roundabout, whirligig',
 477: "carpenter's kit, tool kit",
 478: 'carton',
 479: 'car wheel',
 480: 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM',
 481: 'cassette',
 482: 'cassette player',
 483: 'castle',
 484: 'catamaran',
 485: 'CD player',
 486: 'cello, violoncello',
 487: 'cellular telephone, cellular phone, cellphone, cell, mobile phone',
 488: 'chain',
 489: 'chainlink fence',
 490: 'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour',
 491: 'chain saw, chainsaw',
 492: 'chest',
 493: 'chiffonier, commode',
 494: 'chime, bell, gong',
 495: 'china cabinet, china closet',
 496: 'Christmas stocking',
 497: 'church, church building',
 498: 'cinema, movie theater, movie theatre, movie house, picture palace',
 499: 'cleaver, meat cleaver, chopper',
 500: 'cliff dwelling',
 501: 'cloak',
 502: 'clog, geta, patten, sabot',
 503: 'cocktail shaker',
 504: 'coffee mug',
 505: 'coffeepot',
 506: 'coil, spiral, volute, whorl, helix',
 507: 'combination lock',
 508: 'computer keyboard, keypad',
 509: 'confectionery, confectionary, candy store',
 510: 'container ship, containership, container vessel',
 511: 'convertible',
 512: 'corkscrew, bottle screw',
 513: 'cornet, horn, trumpet, trump',
 514: 'cowboy boot',
 515: 'cowboy hat, ten-gallon hat',
 516: 'cradle',
 517: 'crane',
 518: 'crash helmet',
 519: 'crate',
 520: 'crib, cot',
 521: 'Crock Pot',
 522: 'croquet ball',
 523: 'crutch',
 524: 'cuirass',
 525: 'dam, dike, dyke',
 526: 'desk',
 527: 'desktop computer',
 528: 'dial telephone, dial phone',
 529: 'diaper, nappy, napkin',
 530: 'digital clock',
 531: 'digital watch',
 532: 'dining table, board',
 533: 'dishrag, dishcloth',
 534: 'dishwasher, dish washer, dishwashing machine',
 535: 'disk brake, disc brake',
 536: 'dock, dockage, docking facility',
 537: 'dogsled, dog sled, dog sleigh',
 538: 'dome',
 539: 'doormat, welcome mat',
 540: 'drilling platform, offshore rig',
 541: 'drum, membranophone, tympan',
 542: 'drumstick',
 543: 'dumbbell',
 544: 'Dutch oven',
 545: 'electric fan, blower',
 546: 'electric guitar',
 547: 'electric locomotive',
 548: 'entertainment center',
 549: 'envelope',
 550: 'espresso maker',
 551: 'face powder',
 552: 'feather boa, boa',
 553: 'file, file cabinet, filing cabinet',
 554: 'fireboat',
 555: 'fire engine, fire truck',
 556: 'fire screen, fireguard',
 557: 'flagpole, flagstaff',
 558: 'flute, transverse flute',
 559: 'folding chair',
 560: 'football helmet',
 561: 'forklift',
 562: 'fountain',
 563: 'fountain pen',
 564: 'four-poster',
 565: 'freight car',
 566: 'French horn, horn',
 567: 'frying pan, frypan, skillet',
 568: 'fur coat',
 569: 'garbage truck, dustcart',
 570: 'gasmask, respirator, gas helmet',
 571: 'gas pump, gasoline pump, petrol pump, island dispenser',
 572: 'goblet',
 573: 'go-kart',
 574: 'golf ball',
 575: 'golfcart, golf cart',
 576: 'gondola',
 577: 'gong, tam-tam',
 578: 'gown',
 579: 'grand piano, grand',
 580: 'greenhouse, nursery, glasshouse',
 581: 'grille, radiator grille',
 582: 'grocery store, grocery, food market, market',
 583: 'guillotine',
 584: 'hair slide',
 585: 'hair spray',
 586: 'half track',
 587: 'hammer',
 588: 'hamper',
 589: 'hand blower, blow dryer, blow drier, hair dryer, hair drier',
 590: 'hand-held computer, hand-held microcomputer',
 591: 'handkerchief, hankie, hanky, hankey',
 592: 'hard disc, hard disk, fixed disk',
 593: 'harmonica, mouth organ, harp, mouth harp',
 594: 'harp',
 595: 'harvester, reaper',
 596: 'hatchet',
 597: 'holster',
 598: 'home theater, home theatre',
 599: 'honeycomb',
 600: 'hook, claw',
 601: 'hoopskirt, crinoline',
 602: 'horizontal bar, high bar',
 603: 'horse cart, horse-cart',
 604: 'hourglass',
 605: 'iPod',
 606: 'iron, smoothing iron',
 607: "jack-o'-lantern",
 608: 'jean, blue jean, denim',
 609: 'jeep, landrover',
 610: 'jersey, T-shirt, tee shirt',
 611: 'jigsaw puzzle',
 612: 'jinrikisha, ricksha, rickshaw',
 613: 'joystick',
 614: 'kimono',
 615: 'knee pad',
 616: 'knot',
 617: 'lab coat, laboratory coat',
 618: 'ladle',
 619: 'lampshade, lamp shade',
 620: 'laptop, laptop computer',
 621: 'lawn mower, mower',
 622: 'lens cap, lens cover',
 623: 'letter opener, paper knife, paperknife',
 624: 'library',
 625: 'lifeboat',
 626: 'lighter, light, igniter, ignitor',
 627: 'limousine, limo',
 628: 'liner, ocean liner',
 629: 'lipstick, lip rouge',
 630: 'Loafer',
 631: 'lotion',
 632: 'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system',
 633: "loupe, jeweler's loupe",
 634: 'lumbermill, sawmill',
 635: 'magnetic compass',
 636: 'mailbag, postbag',
 637: 'mailbox, letter box',
 638: 'maillot',
 639: 'maillot, tank suit',
 640: 'manhole cover',
 641: 'maraca',
 642: 'marimba, xylophone',
 643: 'mask',
 644: 'matchstick',
 645: 'maypole',
 646: 'maze, labyrinth',
 647: 'measuring cup',
 648: 'medicine chest, medicine cabinet',
 649: 'megalith, megalithic structure',
 650: 'microphone, mike',
 651: 'microwave, microwave oven',
 652: 'military uniform',
 653: 'milk can',
 654: 'minibus',
 655: 'miniskirt, mini',
 656: 'minivan',
 657: 'missile',
 658: 'mitten',
 659: 'mixing bowl',
 660: 'mobile home, manufactured home',
 661: 'Model T',
 662: 'modem',
 663: 'monastery',
 664: 'monitor',
 665: 'moped',
 666: 'mortar',
 667: 'mortarboard',
 668: 'mosque',
 669: 'mosquito net',
 670: 'motor scooter, scooter',
 671: 'mountain bike, all-terrain bike, off-roader',
 672: 'mountain tent',
 673: 'mouse, computer mouse',
 674: 'mousetrap',
 675: 'moving van',
 676: 'muzzle',
 677: 'nail',
 678: 'neck brace',
 679: 'necklace',
 680: 'nipple',
 681: 'notebook, notebook computer',
 682: 'obelisk',
 683: 'oboe, hautboy, hautbois',
 684: 'ocarina, sweet potato',
 685: 'odometer, hodometer, mileometer, milometer',
 686: 'oil filter',
 687: 'organ, pipe organ',
 688: 'oscilloscope, scope, cathode-ray oscilloscope, CRO',
 689: 'overskirt',
 690: 'oxcart',
 691: 'oxygen mask',
 692: 'packet',
 693: 'paddle, boat paddle',
 694: 'paddlewheel, paddle wheel',
 695: 'padlock',
 696: 'paintbrush',
 697: "pajama, pyjama, pj's, jammies",
 698: 'palace',
 699: 'panpipe, pandean pipe, syrinx',
 700: 'paper towel',
 701: 'parachute, chute',
 702: 'parallel bars, bars',
 703: 'park bench',
 704: 'parking meter',
 705: 'passenger car, coach, carriage',
 706: 'patio, terrace',
 707: 'pay-phone, pay-station',
 708: 'pedestal, plinth, footstall',
 709: 'pencil box, pencil case',
 710: 'pencil sharpener',
 711: 'perfume, essence',
 712: 'Petri dish',
 713: 'photocopier',
 714: 'pick, plectrum, plectron',
 715: 'pickelhaube',
 716: 'picket fence, paling',
 717: 'pickup, pickup truck',
 718: 'pier',
 719: 'piggy bank, penny bank',
 720: 'pill bottle',
 721: 'pillow',
 722: 'ping-pong ball',
 723: 'pinwheel',
 724: 'pirate, pirate ship',
 725: 'pitcher, ewer',
 726: "plane, carpenter's plane, woodworking plane",
 727: 'planetarium',
 728: 'plastic bag',
 729: 'plate rack',
 730: 'plow, plough',
 731: "plunger, plumber's helper",
 732: 'Polaroid camera, Polaroid Land camera',
 733: 'pole',
 734: 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',
 735: 'poncho',
 736: 'pool table, billiard table, snooker table',
 737: 'pop bottle, soda bottle',
 738: 'pot, flowerpot',
 739: "potter's wheel",
 740: 'power drill',
 741: 'prayer rug, prayer mat',
 742: 'printer',
 743: 'prison, prison house',
 744: 'projectile, missile',
 745: 'projector',
 746: 'puck, hockey puck',
 747: 'punching bag, punch bag, punching ball, punchball',
 748: 'purse',
 749: 'quill, quill pen',
 750: 'quilt, comforter, comfort, puff',
 751: 'racer, race car, racing car',
 752: 'racket, racquet',
 753: 'radiator',
 754: 'radio, wireless',
 755: 'radio telescope, radio reflector',
 756: 'rain barrel',
 757: 'recreational vehicle, RV, R.V.',
 758: 'reel',
 759: 'reflex camera',
 760: 'refrigerator, icebox',
 761: 'remote control, remote',
 762: 'restaurant, eating house, eating place, eatery',
 763: 'revolver, six-gun, six-shooter',
 764: 'rifle',
 765: 'rocking chair, rocker',
 766: 'rotisserie',
 767: 'rubber eraser, rubber, pencil eraser',
 768: 'rugby ball',
 769: 'rule, ruler',
 770: 'running shoe',
 771: 'safe',
 772: 'safety pin',
 773: 'saltshaker, salt shaker',
 774: 'sandal',
 775: 'sarong',
 776: 'sax, saxophone',
 777: 'scabbard',
 778: 'scale, weighing machine',
 779: 'school bus',
 780: 'schooner',
 781: 'scoreboard',
 782: 'screen, CRT screen',
 783: 'screw',
 784: 'screwdriver',
 785: 'seat belt, seatbelt',
 786: 'sewing machine',
 787: 'shield, buckler',
 788: 'shoe shop, shoe-shop, shoe store',
 789: 'shoji',
 790: 'shopping basket',
 791: 'shopping cart',
 792: 'shovel',
 793: 'shower cap',
 794: 'shower curtain',
 795: 'ski',
 796: 'ski mask',
 797: 'sleeping bag',
 798: 'slide rule, slipstick',
 799: 'sliding door',
 800: 'slot, one-armed bandit',
 801: 'snorkel',
 802: 'snowmobile',
 803: 'snowplow, snowplough',
 804: 'soap dispenser',
 805: 'soccer ball',
 806: 'sock',
 807: 'solar dish, solar collector, solar furnace',
 808: 'sombrero',
 809: 'soup bowl',
 810: 'space bar',
 811: 'space heater',
 812: 'space shuttle',
 813: 'spatula',
 814: 'speedboat',
 815: "spider web, spider's web",
 816: 'spindle',
 817: 'sports car, sport car',
 818: 'spotlight, spot',
 819: 'stage',
 820: 'steam locomotive',
 821: 'steel arch bridge',
 822: 'steel drum',
 823: 'stethoscope',
 824: 'stole',
 825: 'stone wall',
 826: 'stopwatch, stop watch',
 827: 'stove',
 828: 'strainer',
 829: 'streetcar, tram, tramcar, trolley, trolley car',
 830: 'stretcher',
 831: 'studio couch, day bed',
 832: 'stupa, tope',
 833: 'submarine, pigboat, sub, U-boat',
 834: 'suit, suit of clothes',
 835: 'sundial',
 836: 'sunglass',
 837: 'sunglasses, dark glasses, shades',
 838: 'sunscreen, sunblock, sun blocker',
 839: 'suspension bridge',
 840: 'swab, swob, mop',
 841: 'sweatshirt',
 842: 'swimming trunks, bathing trunks',
 843: 'swing',
 844: 'switch, electric switch, electrical switch',
 845: 'syringe',
 846: 'table lamp',
 847: 'tank, army tank, armored combat vehicle, armoured combat vehicle',
 848: 'tape player',
 849: 'teapot',
 850: 'teddy, teddy bear',
 851: 'television, television system',
 852: 'tennis ball',
 853: 'thatch, thatched roof',
 854: 'theater curtain, theatre curtain',
 855: 'thimble',
 856: 'thresher, thrasher, threshing machine',
 857: 'throne',
 858: 'tile roof',
 859: 'toaster',
 860: 'tobacco shop, tobacconist shop, tobacconist',
 861: 'toilet seat',
 862: 'torch',
 863: 'totem pole',
 864: 'tow truck, tow car, wrecker',
 865: 'toyshop',
 866: 'tractor',
 867: 'trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi',
 868: 'tray',
 869: 'trench coat',
 870: 'tricycle, trike, velocipede',
 871: 'trimaran',
 872: 'tripod',
 873: 'triumphal arch',
 874: 'trolleybus, trolley coach, trackless trolley',
 875: 'trombone',
 876: 'tub, vat',
 877: 'turnstile',
 878: 'typewriter keyboard',
 879: 'umbrella',
 880: 'unicycle, monocycle',
 881: 'upright, upright piano',
 882: 'vacuum, vacuum cleaner',
 883: 'vase',
 884: 'vault',
 885: 'velvet',
 886: 'vending machine',
 887: 'vestment',
 888: 'viaduct',
 889: 'violin, fiddle',
 890: 'volleyball',
 891: 'waffle iron',
 892: 'wall clock',
 893: 'wallet, billfold, notecase, pocketbook',
 894: 'wardrobe, closet, press',
 895: 'warplane, military plane',
 896: 'washbasin, handbasin, washbowl, lavabo, wash-hand basin',
 897: 'washer, automatic washer, washing machine',
 898: 'water bottle',
 899: 'water jug',
 900: 'water tower',
 901: 'whiskey jug',
 902: 'whistle',
 903: 'wig',
 904: 'window screen',
 905: 'window shade',
 906: 'Windsor tie',
 907: 'wine bottle',
 908: 'wing',
 909: 'wok',
 910: 'wooden spoon',
 911: 'wool, woolen, woollen',
 912: 'worm fence, snake fence, snake-rail fence, Virginia fence',
 913: 'wreck',
 914: 'yawl',
 915: 'yurt',
 916: 'web site, website, internet site, site',
 917: 'comic book',
 918: 'crossword puzzle, crossword',
 919: 'street sign',
 920: 'traffic light, traffic signal, stoplight',
 921: 'book jacket, dust cover, dust jacket, dust wrapper',
 922: 'menu',
 923: 'plate',
 924: 'guacamole',
 925: 'consomme',
 926: 'hot pot, hotpot',
 927: 'trifle',
 928: 'ice cream, icecream',
 929: 'ice lolly, lolly, lollipop, popsicle',
 930: 'French loaf',
 931: 'bagel, beigel',
 932: 'pretzel',
 933: 'cheeseburger',
 934: 'hotdog, hot dog, red hot',
 935: 'mashed potato',
 936: 'head cabbage',
 937: 'broccoli',
 938: 'cauliflower',
 939: 'zucchini, courgette',
 940: 'spaghetti squash',
 941: 'acorn squash',
 942: 'butternut squash',
 943: 'cucumber, cuke',
 944: 'artichoke, globe artichoke',
 945: 'bell pepper',
 946: 'cardoon',
 947: 'mushroom',
 948: 'Granny Smith',
 949: 'strawberry',
 950: 'orange',
 951: 'lemon',
 952: 'fig',
 953: 'pineapple, ananas',
 954: 'banana',
 955: 'jackfruit, jak, jack',
 956: 'custard apple',
 957: 'pomegranate',
 958: 'hay',
 959: 'carbonara',
 960: 'chocolate sauce, chocolate syrup',
 961: 'dough',
 962: 'meat loaf, meatloaf',
 963: 'pizza, pizza pie',
 964: 'potpie',
 965: 'burrito',
 966: 'red wine',
 967: 'espresso',
 968: 'cup',
 969: 'eggnog',
 970: 'alp',
 971: 'bubble',
 972: 'cliff, drop, drop-off',
 973: 'coral reef',
 974: 'geyser',
 975: 'lakeside, lakeshore',
 976: 'promontory, headland, head, foreland',
 977: 'sandbar, sand bar',
 978: 'seashore, coast, seacoast, sea-coast',
 979: 'valley, vale',
 980: 'volcano',
 981: 'ballplayer, baseball player',
 982: 'groom, bridegroom',
 983: 'scuba diver',
 984: 'rapeseed',
 985: 'daisy',
 986: "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum",
 987: 'corn',
 988: 'acorn',
 989: 'hip, rose hip, rosehip',
 990: 'buckeye, horse chestnut, conker',
 991: 'coral fungus',
 992: 'agaric',
 993: 'gyromitra',
 994: 'stinkhorn, carrion fungus',
 995: 'earthstar',
 996: 'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa',
 997: 'bolete',
 998: 'ear, spike, capitulum',
 999: 'toilet tissue, toilet paper, bathroom tissue'
}

ddim_alphas = torch.tensor([0.9991, 0.9983, 0.9974, 0.9966, 0.9957, 0.9948, 0.9940, 0.9931, 0.9922,
        0.9913, 0.9904, 0.9895, 0.9886, 0.9877, 0.9868, 0.9859, 0.9850, 0.9841,
        0.9832, 0.9822, 0.9813, 0.9804, 0.9794, 0.9785, 0.9776, 0.9766, 0.9757,
        0.9747, 0.9737, 0.9728, 0.9718, 0.9708, 0.9698, 0.9689, 0.9679, 0.9669,
        0.9659, 0.9649, 0.9639, 0.9629, 0.9619, 0.9609, 0.9599, 0.9588, 0.9578,
        0.9568, 0.9557, 0.9547, 0.9537, 0.9526, 0.9516, 0.9505, 0.9495, 0.9484,
        0.9473, 0.9463, 0.9452, 0.9441, 0.9430, 0.9420, 0.9409, 0.9398, 0.9387,
        0.9376, 0.9365, 0.9354, 0.9343, 0.9332, 0.9320, 0.9309, 0.9298, 0.9287,
        0.9275, 0.9264, 0.9252, 0.9241, 0.9229, 0.9218, 0.9206, 0.9195, 0.9183,
        0.9171, 0.9160, 0.9148, 0.9136, 0.9124, 0.9112, 0.9100, 0.9089, 0.9077,
        0.9065, 0.9052, 0.9040, 0.9028, 0.9016, 0.9004, 0.8992, 0.8979, 0.8967,
        0.8955, 0.8942, 0.8930, 0.8917, 0.8905, 0.8892, 0.8880, 0.8867, 0.8854,
        0.8842, 0.8829, 0.8816, 0.8804, 0.8791, 0.8778, 0.8765, 0.8752, 0.8739,
        0.8726, 0.8713, 0.8700, 0.8687, 0.8674, 0.8661, 0.8647, 0.8634, 0.8621,
        0.8607, 0.8594, 0.8581, 0.8567, 0.8554, 0.8540, 0.8527, 0.8513, 0.8500,
        0.8486, 0.8473, 0.8459, 0.8445, 0.8431, 0.8418, 0.8404, 0.8390, 0.8376,
        0.8362, 0.8348, 0.8334, 0.8320, 0.8306, 0.8292, 0.8278, 0.8264, 0.8250,
        0.8236, 0.8221, 0.8207, 0.8193, 0.8179, 0.8164, 0.8150, 0.8136, 0.8121,
        0.8107, 0.8092, 0.8078, 0.8063, 0.8049, 0.8034, 0.8019, 0.8005, 0.7990,
        0.7975, 0.7960, 0.7946, 0.7931, 0.7916, 0.7901, 0.7886, 0.7871, 0.7856,
        0.7842, 0.7827, 0.7812, 0.7796, 0.7781, 0.7766, 0.7751, 0.7736, 0.7721,
        0.7706, 0.7690, 0.7675, 0.7660, 0.7645, 0.7629, 0.7614, 0.7599, 0.7583,
        0.7568, 0.7552, 0.7537, 0.7521, 0.7506, 0.7490, 0.7475, 0.7459, 0.7444,
        0.7428, 0.7412, 0.7397, 0.7381, 0.7365, 0.7350, 0.7334, 0.7318, 0.7302,
        0.7286, 0.7271, 0.7255, 0.7239, 0.7223, 0.7207, 0.7191, 0.7175, 0.7159,
        0.7143, 0.7127, 0.7111, 0.7095, 0.7079, 0.7063, 0.7047, 0.7031, 0.7015,
        0.6999, 0.6982, 0.6966, 0.6950, 0.6934, 0.6918, 0.6901, 0.6885, 0.6869,
        0.6852, 0.6836, 0.6820, 0.6803, 0.6787, 0.6771, 0.6754, 0.6738, 0.6722,
        0.6705, 0.6689, 0.6672, 0.6656, 0.6639, 0.6623, 0.6606, 0.6590, 0.6573,
        0.6557, 0.6540, 0.6524, 0.6507, 0.6490, 0.6474, 0.6457, 0.6441, 0.6424,
        0.6407, 0.6391, 0.6374, 0.6357, 0.6341, 0.6324, 0.6307, 0.6291, 0.6274,
        0.6257, 0.6241, 0.6224, 0.6207, 0.6190, 0.6174, 0.6157, 0.6140, 0.6123,
        0.6107, 0.6090, 0.6073, 0.6056, 0.6039, 0.6023, 0.6006, 0.5989, 0.5972,
        0.5955, 0.5939, 0.5922, 0.5905, 0.5888, 0.5871, 0.5855, 0.5838, 0.5821,
        0.5804, 0.5787, 0.5770, 0.5754, 0.5737, 0.5720, 0.5703, 0.5686, 0.5669,
        0.5652, 0.5636, 0.5619, 0.5602, 0.5585, 0.5568, 0.5551, 0.5535, 0.5518,
        0.5501, 0.5484, 0.5467, 0.5450, 0.5434, 0.5417, 0.5400, 0.5383, 0.5366,
        0.5350, 0.5333, 0.5316, 0.5299, 0.5282, 0.5266, 0.5249, 0.5232, 0.5215,
        0.5199, 0.5182, 0.5165, 0.5148, 0.5132, 0.5115, 0.5098, 0.5082, 0.5065,
        0.5048, 0.5032, 0.5015, 0.4998, 0.4982, 0.4965, 0.4948, 0.4932, 0.4915,
        0.4898, 0.4882, 0.4865, 0.4849, 0.4832, 0.4816, 0.4799, 0.4782, 0.4766,
        0.4749, 0.4733, 0.4716, 0.4700, 0.4684, 0.4667, 0.4651, 0.4634, 0.4618,
        0.4601, 0.4585, 0.4569, 0.4552, 0.4536, 0.4520, 0.4503, 0.4487, 0.4471,
        0.4455, 0.4438, 0.4422, 0.4406, 0.4390, 0.4374, 0.4357, 0.4341, 0.4325,
        0.4309, 0.4293, 0.4277, 0.4261, 0.4245, 0.4229, 0.4213, 0.4197, 0.4181,
        0.4165, 0.4149, 0.4133, 0.4117, 0.4101, 0.4086, 0.4070, 0.4054, 0.4038,
        0.4022, 0.4007, 0.3991, 0.3975, 0.3960, 0.3944, 0.3928, 0.3913, 0.3897,
        0.3882, 0.3866, 0.3850, 0.3835, 0.3819, 0.3804, 0.3789, 0.3773, 0.3758,
        0.3742, 0.3727, 0.3712, 0.3697, 0.3681, 0.3666, 0.3651, 0.3636, 0.3621,
        0.3605, 0.3590, 0.3575, 0.3560, 0.3545, 0.3530, 0.3515, 0.3500, 0.3485,
        0.3470, 0.3456, 0.3441, 0.3426, 0.3411, 0.3396, 0.3382, 0.3367, 0.3352,
        0.3338, 0.3323, 0.3308, 0.3294, 0.3279, 0.3265, 0.3250, 0.3236, 0.3222,
        0.3207, 0.3193, 0.3178, 0.3164, 0.3150, 0.3136, 0.3122, 0.3107, 0.3093,
        0.3079, 0.3065, 0.3051, 0.3037, 0.3023, 0.3009, 0.2995, 0.2981, 0.2967,
        0.2954, 0.2940, 0.2926, 0.2912, 0.2899, 0.2885, 0.2871, 0.2858, 0.2844,
        0.2831, 0.2817, 0.2804, 0.2790, 0.2777, 0.2763, 0.2750, 0.2737, 0.2723,
        0.2710, 0.2697, 0.2684, 0.2671, 0.2658, 0.2645, 0.2631, 0.2618, 0.2606,
        0.2593, 0.2580, 0.2567, 0.2554, 0.2541, 0.2528, 0.2516, 0.2503, 0.2490,
        0.2478, 0.2465, 0.2453, 0.2440, 0.2428, 0.2415, 0.2403, 0.2391, 0.2378,
        0.2366, 0.2354, 0.2341, 0.2329, 0.2317, 0.2305, 0.2293, 0.2281, 0.2269,
        0.2257, 0.2245, 0.2233, 0.2221, 0.2209, 0.2198, 0.2186, 0.2174, 0.2163,
        0.2151, 0.2139, 0.2128, 0.2116, 0.2105, 0.2093, 0.2082, 0.2071, 0.2059,
        0.2048, 0.2037, 0.2026, 0.2014, 0.2003, 0.1992, 0.1981, 0.1970, 0.1959,
        0.1948, 0.1937, 0.1926, 0.1915, 0.1905, 0.1894, 0.1883, 0.1872, 0.1862,
        0.1851, 0.1841, 0.1830, 0.1820, 0.1809, 0.1799, 0.1788, 0.1778, 0.1768,
        0.1757, 0.1747, 0.1737, 0.1727, 0.1717, 0.1707, 0.1696, 0.1686, 0.1677,
        0.1667, 0.1657, 0.1647, 0.1637, 0.1627, 0.1618, 0.1608, 0.1598, 0.1589,
        0.1579, 0.1569, 0.1560, 0.1550, 0.1541, 0.1532, 0.1522, 0.1513, 0.1504,
        0.1494, 0.1485, 0.1476, 0.1467, 0.1458, 0.1449, 0.1440, 0.1431, 0.1422,
        0.1413, 0.1404, 0.1395, 0.1386, 0.1378, 0.1369, 0.1360, 0.1352, 0.1343,
        0.1334, 0.1326, 0.1317, 0.1309, 0.1301, 0.1292, 0.1284, 0.1276, 0.1267,
        0.1259, 0.1251, 0.1243, 0.1235, 0.1227, 0.1219, 0.1211, 0.1203, 0.1195,
        0.1187, 0.1179, 0.1171, 0.1163, 0.1155, 0.1148, 0.1140, 0.1132, 0.1125,
        0.1117, 0.1110, 0.1102, 0.1095, 0.1087, 0.1080, 0.1073, 0.1065, 0.1058,
        0.1051, 0.1044, 0.1036, 0.1029, 0.1022, 0.1015, 0.1008, 0.1001, 0.0994,
        0.0987, 0.0980, 0.0973, 0.0967, 0.0960, 0.0953, 0.0946, 0.0940, 0.0933,
        0.0926, 0.0920, 0.0913, 0.0907, 0.0900, 0.0894, 0.0887, 0.0881, 0.0875,
        0.0868, 0.0862, 0.0856, 0.0850, 0.0844, 0.0837, 0.0831, 0.0825, 0.0819,
        0.0813, 0.0807, 0.0801, 0.0795, 0.0789, 0.0784, 0.0778, 0.0772, 0.0766,
        0.0761, 0.0755, 0.0749, 0.0744, 0.0738, 0.0732, 0.0727, 0.0721, 0.0716,
        0.0711, 0.0705, 0.0700, 0.0694, 0.0689, 0.0684, 0.0679, 0.0673, 0.0668,
        0.0663, 0.0658, 0.0653, 0.0648, 0.0643, 0.0638, 0.0633, 0.0628, 0.0623,
        0.0618, 0.0613, 0.0608, 0.0604, 0.0599, 0.0594, 0.0589, 0.0585, 0.0580,
        0.0575, 0.0571, 0.0566, 0.0562, 0.0557, 0.0553, 0.0548, 0.0544, 0.0539,
        0.0535, 0.0531, 0.0526, 0.0522, 0.0518, 0.0514, 0.0509, 0.0505, 0.0501,
        0.0497, 0.0493, 0.0489, 0.0485, 0.0481, 0.0477, 0.0473, 0.0469, 0.0465,
        0.0461, 0.0457, 0.0453, 0.0450, 0.0446, 0.0442, 0.0438, 0.0435, 0.0431,
        0.0427, 0.0424, 0.0420, 0.0416, 0.0413, 0.0409, 0.0406, 0.0402, 0.0399,
        0.0395, 0.0392, 0.0389, 0.0385, 0.0382, 0.0379, 0.0375, 0.0372, 0.0369,
        0.0365, 0.0362, 0.0359, 0.0356, 0.0353, 0.0350, 0.0347, 0.0343, 0.0340,
        0.0337, 0.0334, 0.0331, 0.0328, 0.0325, 0.0323, 0.0320, 0.0317, 0.0314,
        0.0311, 0.0308, 0.0305, 0.0303, 0.0300, 0.0297, 0.0295, 0.0292, 0.0289,
        0.0286, 0.0284, 0.0281, 0.0279, 0.0276, 0.0274, 0.0271, 0.0268, 0.0266,
        0.0264, 0.0261, 0.0259, 0.0256, 0.0254, 0.0251, 0.0249, 0.0247, 0.0244,
        0.0242, 0.0240, 0.0237, 0.0235, 0.0233, 0.0231, 0.0229, 0.0226, 0.0224,
        0.0222, 0.0220, 0.0218, 0.0216, 0.0214, 0.0212, 0.0210, 0.0207, 0.0205,
        0.0203, 0.0201, 0.0200, 0.0198, 0.0196, 0.0194, 0.0192, 0.0190, 0.0188,
        0.0186, 0.0184, 0.0182, 0.0181, 0.0179, 0.0177, 0.0175, 0.0174, 0.0172,
        0.0170, 0.0168, 0.0167, 0.0165, 0.0163, 0.0162, 0.0160, 0.0158, 0.0157,
        0.0155, 0.0154, 0.0152, 0.0151, 0.0149, 0.0147, 0.0146, 0.0144, 0.0143,
        0.0142, 0.0140, 0.0139, 0.0137, 0.0136, 0.0134, 0.0133, 0.0132, 0.0130,
        0.0129, 0.0127, 0.0126, 0.0125, 0.0123, 0.0122, 0.0121, 0.0120, 0.0118,
        0.0117, 0.0116, 0.0115, 0.0113, 0.0112, 0.0111, 0.0110, 0.0109, 0.0107,
        0.0106, 0.0105, 0.0104, 0.0103, 0.0102, 0.0101, 0.0100, 0.0098, 0.0097,
        0.0096, 0.0095, 0.0094, 0.0093, 0.0092, 0.0091, 0.0090, 0.0089, 0.0088,
        0.0087, 0.0086, 0.0085, 0.0084, 0.0083, 0.0082, 0.0082, 0.0081, 0.0080,
        0.0079, 0.0078, 0.0077, 0.0076, 0.0075, 0.0074, 0.0074, 0.0073, 0.0072,
        0.0071, 0.0070, 0.0070, 0.0069, 0.0068, 0.0067, 0.0066, 0.0066, 0.0065,
        0.0064, 0.0063, 0.0063, 0.0062, 0.0061, 0.0061, 0.0060, 0.0059, 0.0058,
        0.0058, 0.0057, 0.0056, 0.0056, 0.0055, 0.0054, 0.0054, 0.0053, 0.0053,
        0.0052, 0.0051, 0.0051, 0.0050, 0.0049, 0.0049, 0.0048, 0.0048, 0.0047,
        0.0047])

