import argparse
import os
import torch
import sys


parser = argparse.ArgumentParser()

# ==================== new defined ================== #
parser.add_argument('--target_smiles', default='')
parser.add_argument('--evaluator', type=str)
parser.add_argument('--reaction_generators', type=str)
parser.add_argument('--reaction_checker', type=str)
parser.add_argument('--max_iteration', type=int, default=20)
parser.add_argument('--building_block', type=str, default='resource/n1-stock.txt')

common_args = parser.parse_args()

