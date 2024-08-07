from rdkit import Chem
from IPython import embed

import traceback

class Parameters:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class Reaction_Database_Node:
    def __init__(self, smiles):
        self.smiles = smiles
        self.reactions = None
        self.best_child = {}

class Reaction_Database:
    def __init__(self, reaction_systems, check_systems = []):
        self.reaction_center = {}
        self.reaction_systems = reaction_systems
        self.check_systems = check_systems
        self.fail = None
        self.run_reverse_count = 0

    def get(self, smiles):
        if smiles not in self.reaction_center:
            self.reaction_center[smiles] = Reaction_Database_Node(smiles)
        return self.reaction_center[smiles]
    
    def set_first_step(self, target_smiles, reaction_json):
        import json
        node = self.get(target_smiles)
        reaction_infor = json.loads(reaction_json)
        from reaction_generator.reaction_generator import Reaction
        reaction = Reaction(reaction_infor['smiles'].split('>')[0].split('.'), target_smiles, Condition(), 1, reaction_infor['probability'], reaction_infor['confidence_score'], True)
        reaction.reaction_type = reaction_infor['reaction_type']
        node.reactions = [reaction]

    def expand(self, smiles):
        """
            Expand reactions of node
        """
        node = self.get(smiles)
        reactions = []
        for reaction_system in self.reaction_systems:
            reactions.extend(reaction_system.generate(smiles))
        for check_system in self.check_systems:
            reactions = check_system.check(reactions)

        for reaction in reactions:
            repeat = False
            for i, rxn in enumerate(reactions):
                if set(rxn.precursors).issubset(set(reaction.precursors)):
                    repeat = True
                    if set(rxn.precursors) == set(reaction.precursors):
                        if reaction.reaction_cost < rxn.reaction_cost:
                            reactions[i] = reaction
                            break
            if not repeat and reaction.product not in reaction.precursors:
                reactions.append(reaction)
        node.reactions = reactions

    def __str__(self) -> str:
        return f"{self.hash}, cost: {self.reaction_cost}, prob: {self.probability}"
    
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Reaction):
            if set(self.precursors) != set(__value.precursors):
                return False
            if set(self.product) != set(__value.product):
                return False
            return True
        return False

class Path_Node:

    @classmethod
    def set_database(cls, database):
        cls.database = database

    @classmethod
    def set_value_fn(cls, value_fn = None):
        if value_fn == 'ours':
            from search_strategy.evaluate import evaluate5 as evaluate_our_rules
            cls.value_fn = evaluate_our_rules
        elif value_fn == 'retro':
            from search_strategy.value_fn_retro import value_fn as evaluate_retro
            cls.value_fn = evaluate_retro
        elif value_fn == 'bart':
            from search_strategy.value_fn_bart import evaluate as evaluate_bart
            cls.value_fn = evaluate_bart
        else:
            cls.value_fn = lambda x: 0

    def __init__(self, molecules, current_cost, parent):
        molecules.sort(key=lambda x:(-len(x),x))
        self.molecules = molecules
        self.current_cost = current_cost    # init as 0
        self.best_cost = None
        self.hash = ' '.join(sorted(molecules))
        self.children = None
        self.estimate_cost = 0
        
        start_time = time.time()
        for smiles in molecules:
            if smiles in Path_Node.database.reaction_center:
                if Path_Node.database.reaction_center[smiles].reactions == []:
                    self.estimate_cost += 1e10   
        self.estimate_cost += Path_Node.value_fn(smiles)
        
        self.parent = parent 

    def __hash__(self):
        return self.hash.__hash__()

    def __eq__(self, obj2):
        if isinstance(obj2, self.__class__):
            return self.hash == obj2.hash
        else:
            return False

    def __lt__(self, obj2):
        if isinstance(obj2, self.__class__):
            return self.hash < obj2.hash
        else:
            return False

class Synthesis_Path:

    def __init__(self, ):
        self.reactions = []
        self.materials = {}
        self.reactions_type = []
        self.run_reverse_count = 0

    def to_json(self, ):
        import json
        return json.dumps({
            'reactions':[rxn.to_json() for rxn in self.reactions], 
            'steps':len(self.reactions), 
            'materials': self.materials,
            'difficulty': self.difficulty,
            'time': self.time, 
            'eval_time': self.eval_time, 
            'search_time': self.time - self.eval_time,
            'run_reverse_count': self.run_reverse_count,
            'backup_reactions':{rxn.product:[rxn2.to_json() for rxn2 in self.backup_reactions[rxn.product].reactions] for rxn in self.reactions}
            }, indent = 4)

