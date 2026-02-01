import csv 

from train_methods.consts import IMAGENET_1K


class ConceptDict:
    def __init__(self):
        self.all_concepts: dict[str, list] = {}

    def load_concepts(self, concept_name: str, csv_file_path: str):
        
        data = []
        with open(csv_file_path, mode='r') as file:
            reader = csv.reader(file)
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
