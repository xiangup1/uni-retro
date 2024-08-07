from search_strategies.basic import Reaction_Database, Parameters
from search_strategies.parse_args import common_args
import threading
from reaction_generator.basic import loadGenerator
from reaction_checker.basic import loadChecker
from evaluator.basic import loadEvaluator
from building_block import loadBuildingBlock
from pipeline_utils import simplify
from queue import PriorityQueue, Queue
import sys
import json
import os
import time
from math import log10
from IPython import embed
import json
import datetime

class Route:

    @classmethod
    def set_database(cls, database):
        cls.database = database

    def __init__(self, nodes=[], cost=0, reactions=[], materials={}):
        self.cost = cost
        self.nodes = nodes
        self.reactions = reactions
        self.materials = materials
    
    def to_json(self,):
        return json.dumps({
            'complete':len(self.nodes)==0,
            'reactions':[rxn.smiles for rxn in self.reactions[::-1]], 
            'steps':len(self.reactions), 
            'materials': self.materials,
            }, indent = 4)


    def __lt__(self, route2):
        return self.cost < route2.cost

class Hope_Tree_Node:
    max_route = 10
    def __init__(self, smiles, estimate_cost = None, root = False):
        self.smiles = smiles
        self.routes = []
        self.complete_routes = []
        self.parents = []
        self.children = None
        self.estimate_cost = estimate_cost
        self.history_routes = [Route(reactions=[], cost=self.estimate_cost, materials=[], nodes=[smiles])]
        self.cost = 0
        price = self.materials_db.get(smiles)
        if not root and price is not None:
            self.cost += log10(price+1)
            self.routes = [Route(materials={smiles:price}, cost=self.cost)]
            self.complete_routes = [Route(materials={smiles:price})]
        else:
            self.cost += self.estimate_cost
            self.routes = [Route(nodes=[smiles], cost=self.cost)]
    
    @classmethod
    def set_database(cls, database):
        cls.database = database

    @classmethod
    def set_materials(cls, materials_db):
        cls.materials_db = materials_db

def nodes_and(routes1, routes2, ):
    cost_route = []
    for idx1, route1 in enumerate(routes1):
        for idx2, route2 in enumerate(routes2):
            cost_route.append((route1.cost + route2.cost, idx1, idx2))
    cost_route.sort()
    routes = []
    for cost, idx1, idx2 in cost_route:
        if len(routes) == Hope_Tree_Node.max_route and cost > routes[-1][0]:
            continue
        route1 = routes1[idx1]
        route2 = routes2[idx2]
        new_route = Route()
        new_route.cost = route1.cost + route2.cost
        new_route.nodes = route1.nodes + route2.nodes
        new_route.reactions = route1.reactions + route2.reactions
        new_route.materials = route1.materials.copy()
        new_route.materials.update(route2.materials)
        circle = False
        for rxn in new_route.reactions:
            if new_route.reactions[0].product in rxn.precursors:
                circle = True
        if not circle:
            max_same = 0
            for _, pre in routes:
                max_same = max(max_same, len(set(pre.reactions) & set(new_route.reactions)))
            if max_same < len(new_route.reactions):
                new_cost = new_route.cost * ((1+max_same / 10) ** 2)
                routes.append((new_cost, new_route))
                for i in range(min(Hope_Tree_Node.max_route-1, len(routes)-2), -1, -1):
                    if new_cost < routes[i][0]:
                        if i != Hope_Tree_Node.max_route-1:
                            routes[i+1] = routes[i]
                        routes[i] = (new_cost, new_route)
                    
                routes = routes[:Hope_Tree_Node.max_route]

    return [route for _, route in routes]

def nodes_or(routes1, routes2):
    origin_routes = routes1 + routes2
    origin_routes.sort()
    routes = []
    for idx, route in enumerate(origin_routes):
        max_same = 0
        for pre in origin_routes[:idx]:
            max_same = max(max_same, len(set(pre.reactions) & set(route.reactions)))
        if max_same < len(route.reactions) or max_same == 0:
            new_cost = route.cost * ((1+max_same / 10) ** 2)
            routes.append((new_cost, route))
            for i in range(min(Hope_Tree_Node.max_route-1, len(routes)-2), -1, -1):
                if new_cost < routes[i][0]:
                    if i != Hope_Tree_Node.max_route-1:
                        routes[i+1] = routes[i]
                    routes[i] = (new_cost, route)                   
            routes = routes[:Hope_Tree_Node.max_route]
    return [route for _, route in routes]

class Hope_Tree:
    def __init__(self, parameter, target_smiles = None):
        self.materials_db = parameter.materials_db
        self.run_time = parameter.run_time
        self.reaction_database = parameter.reaction_database
        self.value_fn = parameter.value_fn
        self.max_iteration = parameter.iterations
        self.lock = threading.Lock()
        Route.set_database(self.reaction_database)
        Hope_Tree_Node.set_database(self.reaction_database)
        Hope_Tree_Node.set_materials(self.materials_db)
        self.root = None
        self.nodes = {}
        if target_smiles is not None:
            self.set_root(target_smiles)     
    
    def set_root(self, target_smiles, root=True):
        self.target_smiles = target_smiles
        estimate_cost = self.value_fn.evaluate([target_smiles])[target_smiles]
        self.root = Hope_Tree_Node(target_smiles, estimate_cost = estimate_cost, root=root)
        self.nodes[target_smiles] = self.root

    def expand(self, smiles):
        self.reaction_database.expand(smiles)
        self.add_reactions(smiles, self.reaction_database.get(smiles).reactions)
    
    def add_reactions(self, parent, reactions):
        self.nodes[parent].children = []
        children = []
        new_precursors = []
        for reaction in reactions:
            materials = {}
            precursors = []
            for precursor in reaction.precursors:
                price = self.materials_db.get(precursor)
                if price is not None:
                    materials[precursor] = price
                else:
                    precursors.append(precursor)
                    if precursor not in self.nodes and precursor not in new_precursors:
                        new_precursors.append(precursor)
            children.append({
                'reaction':reaction,
                'precursors':precursors,
                'materials':materials,
            }
            )
        if new_precursors:
            estimate_cost = self.value_fn.evaluate(new_precursors)
        else:
            estimate_cost = []
        if len(estimate_cost) != len(new_precursors):
            print('value function failed')
            embed()
            return
        children.sort(key=lambda x:(len(x['precursors']), len(x['materials'])))
        for child in children:
            repeat = False
            for ch in self.nodes[parent].children:
                if (ch['precursors'] and set(ch['precursors']).issubset(set(child['precursors']))) or (not ch['precursors'] and set(ch['materials']).issubset(set(child['materials']))):
                    repeat = True
                    break
            for precursor in child['precursors']:
                if precursor not in self.nodes:
                    self.nodes[precursor] = Hope_Tree_Node(precursor, estimate_cost.get(precursor, 0))
            if not repeat:
                self.nodes[parent].children.append(child)
        for child in self.nodes[parent].children:
            for precursor in child['precursors']:
                self.nodes[precursor].parents.append(child)
        #self.backward(parent)

    def update_routes(self, smiles):
        routes = []
        node = self.nodes[smiles]
        for rxn in node.children:
            tmp_routes = [Route(reactions=[rxn['reaction']],cost=rxn['reaction'].reaction_cost,materials=rxn['materials'])]
            for precursor in rxn['precursors']:
                tmp_routes = nodes_and(tmp_routes, self.nodes[precursor].routes)
            routes += tmp_routes
        routes = nodes_or(routes, [])
        changed = self.nodes[smiles].routes != routes
        self.nodes[smiles].routes = routes
        return changed

    def update_complete_routes(self, smiles):
        routes = []
        node = self.nodes[smiles]
        for rxn in node.children:
            tmp_routes = [Route(reactions=[rxn['reaction']],cost=rxn['reaction'].reaction_cost,materials=rxn['materials'])]
            for precursor in rxn['precursors']:
                tmp_routes = nodes_and(tmp_routes, self.nodes[precursor].complete_routes)
            routes += tmp_routes
        routes = nodes_or(routes, [])  
        changed = self.nodes[smiles].complete_routes != routes
        self.nodes[smiles].complete_routes = routes    
        return changed
        
    def backward(self, list_smiles):
        end_flag = False
        repeat_num = 0
        while not end_flag:
            repeat_num += 1
            update_sequence = []
            update_queue = Queue()
            children_num = {}
            end_flag = True
            for smiles, node in self.nodes.items():
                if node.routes:
                    cost = node.routes[0].cost
                else:
                    cost = 1e9
                update_sequence.append((cost,smiles))
                children_num[smiles] = 0
                if node.children:
                    for child in node.children:
                        children_num[smiles] += len(child['precursors'])
                if children_num[smiles] == 0:
                    update_queue.put(smiles)
            update_sequence.sort()
            sequence_index = 0
            updated = set()
            pop_sequence = []
            while True:
                if not update_queue.empty():
                    node_smiles = update_queue.get()
                    pop_way = 'queue'
                else:
                    while sequence_index < len(update_sequence) and update_sequence[sequence_index][1] in updated:
                        sequence_index += 1
                    if sequence_index < len(update_sequence):
                        _, node_smiles = update_sequence[sequence_index]
                        sequence_index += 1
                    else:
                        break
                    pop_way = 'sequence'
                updated.add(node_smiles)
                pop_sequence.append(node_smiles)
                node = self.nodes[node_smiles]
                if node.children is not None:
                    self.update_routes(node_smiles) 
                    self.update_complete_routes(node_smiles)
                    for idx in range(len(node.routes)-1,-1,-1):
                        if set(node.routes[idx].nodes) & set(list_smiles):
                            end_flag = False
                            node.routes.pop(idx)
                for rxn in node.parents:
                    product = rxn['reaction'].product
                    if product not in updated:
                        children_num[product] -= 1
                        if children_num[product] == 0:
                            update_queue.put(product)

                        if children_num[product] < 0:
                            print('error')
                            embed()
            if len(updated) != len(update_sequence):
                print('error2')
                embed()
        for smiles in pop_sequence:
            node = self.nodes[smiles]
            for route in node.routes:
                if set(route.nodes) & set(list_smiles):
                    print('error3')
                    embed()

    def search(self, target_smiles = None, progress_file = ''):
        if target_smiles:
            self.set_root(target_smiles)
        for iter in range(self.max_iteration):
            print('iter:',iter)
            expand_nodes = []
            for route in self.root.routes:
                for smiles in route.nodes:
                    if smiles not in expand_nodes:
                        self.expand(smiles)
                        expand_nodes.append(smiles)
            if not expand_nodes:
                break
            if len(self.root.complete_routes)>10 and iter >= 20:
                break
            self.backward(expand_nodes)
        routes = self.root.complete_routes.copy()
        for route in self.root.routes:
            repeat = False
            for r in routes:
                if set(route.reactions).issubset(set(r.reactions)):
                    repeat = True
            if not repeat:
                routes.append(route)
        routes = routes[:20]
        new_routes = []
        for route in routes:
            uni_reactions = set(route.reactions)
            for other_route in new_routes:
                for reaction in other_route.reactions:
                    if reaction in uni_reactions:
                        uni_reactions.remove(reaction)
            if route.reactions and len(uni_reactions) / len(route.reactions)>0.3:
                new_routes.append(route)
        return new_routes

if __name__ == '__main__':
    target_smiles = simplify(common_args.target_smiles)
    reaction_generators_fold = json.loads(common_args.reaction_generators)
    reaction_checkers_fold = json.loads(common_args.reaction_checker)
    evaluator_fold = common_args.evaluator
    building_block = loadBuildingBlock(common_args.building_block)

    reaction_generators = []
    for reaction_generator_fold in reaction_generators_fold:
        reaction_generators.append(loadGenerator(reaction_generator_fold))
    
    reaction_checker = []
    for reaction_checker_fold in reaction_checkers_fold:
        reaction_checker.append(loadChecker(reaction_checker_fold))

    evaluator = loadEvaluator(evaluator_fold)

    para = Parameters(
        materials_db = building_block,
        reaction_database = Reaction_Database(
            reaction_systems = reaction_generators,
            check_systems = reaction_checker,
            ), 
        iterations = common_args.max_iteration,
        run_time = 3600 * 200,
        value_fn = evaluator,
        )
    
    tree = Hope_Tree(para, target_smiles)
    synthesis_path = tree.search()
    with open('path.txt', 'w', encoding="utf-8") as f:
        for path in synthesis_path:
            print(path.to_json())