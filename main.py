import itertools
import logging
import os
from json import loads

from function_factory import FunctionFactory
from dataset_generator import DatasetGenerator
from model import Model


logger = logging.getLogger(__name__)
CONFIG_FILEPATH = "./config.json"


def validate_config(config: dict) -> bool:
    assert "functions" in config, "Key 'functions' was not found in config dictionary"
    assert "hidden_states" in config, "Key 'hidden_states' was not found in config dictionary"
    assert "epochs" in config, "Key 'epochs' was not found in config dictionary"

    assert type(config["functions"]) is list, "Value of 'functions' is not a list"
    assert all(type(x) is dict for x in config["functions"]), "At least one value of 'functions' is not a dict"

    assert all("function_name" in x for x in config["functions"]), "Key 'function_name' not found in one of the functions dictionaries"
    assert all(type(x["function_name"]) is str for x in config["functions"]), "At least one value of 'function_name' is not a string"

    assert all("start_range" in x for x in config["functions"]), "Key 'start_range' not found in one of the functions dictionaries"
    assert all(type(x["start_range"]) is int for x in config["functions"]), "At least one value of 'start_range' is not a integer"

    assert all("end_range" in x for x in config["functions"]), "Key 'end_range' not found in one of the functions dictionaries"
    assert all(type(x["end_range"]) is int for x in config["functions"]), "At least one value of 'end_range' is not a integer"

    assert all("function_step" in x for x in config["functions"]), "Key 'function_step' not found in one of the functions dictionaries"
    assert all(type(x["function_step"]) is int for x in config["functions"]), "At least one value of 'function_step' is not a integer"

    assert type(config["hidden_states"]) is list, "Value of 'hidden_states' is not a list"
    assert all(type(x) is int for x in config["hidden_states"]), "At least one value of 'hidden_states' is not a integer"

    assert type(config["epochs"]) is int, "Value of 'epochs' is not a integer"


if __name__ == "__main__":
    logging.basicConfig(filename="./output/func_approx.log", filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("Start")

    if not os.path.exists(CONFIG_FILEPATH):
        raise FileNotFoundError("config.json file not found")

    with open(CONFIG_FILEPATH, mode='r') as f:
        config = loads(f.read())

    #validate_config(config)

    for function_config, hidden_state in itertools.product(config["functions"], config["hidden_states"]):
        function_name = function_config["function_name"]
        start_range = function_config["start_range"]
        end_range = function_config["end_range"]
        function_step = function_config["function_step"]

        log_message = f"Training new model\nHidden state: {hidden_state}\nFunction '{function_name}'\nFunction step: {function_step}"
        print(log_message)
        logger.info(log_message)
        output_filepath = f"./output/{function_name}_{hidden_state}_result.npy"

        fn = FunctionFactory.create_function(function_name)

        if fn is None:
            logger.warning(f"Fuction {function_name} is not implemented. Skipping it for this execution.")
            continue

        raw_data = fn.generate_data((start_range, end_range), function_step)

        dataset = DatasetGenerator.generate_dataset(raw_data, 0.99, 0.01)

        model = Model(hidden_state)
        model.train(dataset, config["epochs"], output_filepath, len(raw_data))
        logger.info(f"Model trained. Training information saved to file {output_filepath}")

    logger.info("End")
