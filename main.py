import itertools
import logging
import os
from json import loads

from function_factory import FunctionFactory
from dataset_generator import DatasetGenerator
from model_classes.model import Model


logger = logging.getLogger(__name__)
CONFIG_FILEPATH = "./config.json"


def validate_config(config: dict) -> bool:
    # Check if expected keys are present
    assert "functions" in config, "Key 'functions' was not found in config dictionary"
    assert "hidden_states" in config, "Key 'hidden_states' was not found in config dictionary"
    assert "epochs" in config, "Key 'epochs' was not found in config dictionary"
    assert "start_range" in config, "Key 'start_range' was not found in config dictionary"
    assert "end_range" in config, "Key 'end_range' was not found in config dictionary"
    assert "function_step" in config, "Key 'function_step' was not found in config dictionary"

    # Check if values are of the expected type
    assert type(config["functions"]) is list, "Value of 'functions' is not a list"
    assert all(type(x) is str for x in config["functions"]), "At least one value of 'functions' is not a string"
    assert type(config["hidden_states"]) is list, "Value of 'hidden_states' is not a list"
    assert all(type(x) is int for x in config["hidden_states"]), "At least one value of 'hidden_states' is not an integer"
    assert type(config["epochs"]) is int, "Value of 'epochs' is not an integer"
    assert type(config["start_range"]) is int, "Value of 'start_range' is not an integer"
    assert type(config["end_range"]) is int, "Value of 'end_range' is not an integer"
    assert type(config["function_step"]) is float, "Value of 'function_step' is not a float"

    # Check if values are valid
    assert config["epochs"] > 0, "Value of 'epochs' must be greater than 0."
    assert config["end_range"] > config["start_range"], "Value of 'end_range' must be greater than value of 'start_range'."
    assert config["function_step"] < (config["end_range"] - config["start_range"]), "Value of 'function_step' must be lesser than the difference between 'end_range' and 'start_range'."


if __name__ == "__main__":
    logging.basicConfig(filename="./output/func_approx.log", filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("Start")

    if not os.path.exists(CONFIG_FILEPATH):
        raise FileNotFoundError("config.json file not found")

    with open(CONFIG_FILEPATH, mode='r') as f:
        config = loads(f.read())

    validate_config(config)

    start_range = config["start_range"]
    end_range = config["end_range"]
    function_step = config["function_step"]

    for function_name, hidden_state in itertools.product(config["functions"], config["hidden_states"]):
        log_message = f"Training new model\nHidden state: {hidden_state}\nFunction '{function_name}'\nFunction step: {function_step}"
        print(log_message)
        logger.info(log_message)
        output_filepath = f"./output/{function_name}_{hidden_state}_result.npy"

        fn = FunctionFactory.create_function(function_name)

        if fn is None:
            logger.warning(f"Fuction {function_name} is not implemented. Skipping it for this execution.")
            continue

        raw_data = fn.generate_data((start_range, end_range), function_step)

        dataset = DatasetGenerator.generate_dataset(raw_data, 0.75, 0.15)

        model = Model(hidden_state)
        model.train(dataset, config["epochs"], output_filepath, len(raw_data))
        logger.info(f"Model trained. Training information saved to file {output_filepath}")

    logger.info("End")
