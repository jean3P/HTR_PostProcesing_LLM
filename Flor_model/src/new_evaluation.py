import os
import datetime
from config.flor_parameters import HTRFlorConfig
from config.model_config import get_data_generator
from constants import model_outputs, results_outputs, outputs_hdf5, results_outputs_validation
from evaluation.evaluate_model import evaluate_test_data
from utils.flor.network.model import HTRModel
from utils.flor.data import preproc as pp
import cv2


# Step 1: Load the model and evaluate
def load_and_evaluate_model(dataset_name, test_partition, hdf5_file_path, checkpoint_path):
    # Load the data generator for the test set
    dtgen = get_data_generator(hdf5_file_path, test_partition)

    # Initialize the model with architecture and configuration parameters
    model = HTRModel(
        architecture="flor",
        input_size=HTRFlorConfig.INPUT_SIZE,
        vocab_size=dtgen.tokenizer.vocab_size,
        beam_width=HTRFlorConfig.BEAM_WIDTH,
        stop_tolerance=HTRFlorConfig.STOP_TOLERANCE,
        reduce_tolerance=HTRFlorConfig.REDUCE_TOLERANCE,
        reduce_factor=HTRFlorConfig.REDUCE_FACTOR
    )

    # Compile the model before loading the weights
    model.compile(learning_rate=HTRFlorConfig.LEARNING_RATE)

    # Load the model checkpoint
    model.load_checkpoint(target=checkpoint_path)

    # Start timing the evaluation process
    start_time = datetime.datetime.now()

    # Step 2: Predict and evaluate
    output_path = os.path.join(results_outputs_validation, dataset_name, test_partition)

    # Predict the outputs from the test set using the model
    predicts, _ = model.predict(
        x=dtgen.next_test_batch(),
        steps=dtgen.steps['test'],
        ctc_decode=True,
        verbose=1
    )

    # Decode the predicted results into readable text
    predicts = [dtgen.tokenizer.decode(x[0]) for x in predicts]
    ground_truth = [x.decode() for x in dtgen.dataset['test']['gt']]

    # Calculate the total prediction time
    total_time = datetime.datetime.now() - start_time

    # Save the predictions and ground truth to a text file
    with open(os.path.join(output_path, "predict.txt"), "w") as lg:
        for pd, gt in zip(predicts, ground_truth):
            lg.write(f"TE_L {gt}\nTE_P {pd}\n")

    # Create a directory to save the images if it doesn't exist
    image_save_path = os.path.join(output_path, "images")
    os.makedirs(image_save_path, exist_ok=True)

    # Save the first 10 test images with predictions as files
    for i, item in enumerate(dtgen.dataset['test']['dt'][:10]):
        image_filename = os.path.join(image_save_path, f"test_image_{i}.png")
        cv2.imwrite(image_filename, pp.adjust_to_see(item))  # Save image to file
        print(f"Saved: {image_filename}")
        print(f"Ground Truth: {ground_truth[i]}")
        print(f"Prediction: {predicts[i]}\n")

    # Measure total elapsed time for evaluation
    print(f"Evaluation completed in: {total_time}")
    print(f"Results saved to {output_path}")


# Example usage:
dataset_name = "washington"  # Adjust for your dataset
test_partition = "train_100"  # You can use the test partition name
name_model = "model_flor"
hdf5_file_path = os.path.join(outputs_hdf5, dataset_name, f'{dataset_name}_dataset.hdf5')
checkpoint_path = os.path.join(results_outputs, dataset_name, test_partition, name_model, "checkpoint_weights.hdf5")

load_and_evaluate_model(dataset_name, test_partition, hdf5_file_path, checkpoint_path)
