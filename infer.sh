export LOCAL_RANK=-1

python3 -m search_strategies.retro_star --target_smiles="$1" --reaction_generators="[\"resource/nag2g\", \"resource/bert_generator\", \"resource/similarity\"]" --reaction_checker="[\"resource/bert_checker\"]" --evaluator="resource/bert_evaluator" --building_block="resource/n1-stock.txt"
