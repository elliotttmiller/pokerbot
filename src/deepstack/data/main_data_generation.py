"""
Main entrypoint for DeepStack data generation.
Ported from DeepStack Lua main_data_generation.lua.
"""
from deepstack.data.data_generation import generate_training_data

def main():
    # Example arguments
    train_data_count = 1000
    valid_data_count = 200
    data_path = 'data/deepstacked_training/generated/'
    generate_training_data(train_data_count, valid_data_count, data_path)

# Example usage:
# main()
