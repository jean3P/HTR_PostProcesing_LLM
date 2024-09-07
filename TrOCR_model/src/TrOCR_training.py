import os

import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    default_data_collator

from TrOCR_config import ModelConfig, HandleDataTrOCR
from hdf5_handler import HDF5Handler
from utils import metrics_evaluation
from utils.utils import device, clear_cuda_cache


def train_and_save_model(hdf5_file_path, model_save_dir, processor_save_dir, training_config, size_train):
    """
    Train and save the model using HDF5 data.

    :param hdf5_file_path: Path to the HDF5 file (should be a string).
    :param model_save_dir: Directory where the trained model will be saved.
    :param processor_save_dir: Directory where the processor will be saved.
    :param training_config: Configuration for training (batch size, learning rate, etc.).
    :param size_train: The partition of the training dataset to use (e.g., 'train_25', 'train_100').
    """

    # Ensure hdf5_file_path is a string
    if not isinstance(hdf5_file_path, str):
        raise ValueError(f"hdf5_file_path should be a string, but got {type(hdf5_file_path)}")

    # Debug print to verify the value
    print(f"Using HDF5 file path: {hdf5_file_path}")

    # Initialize the processor
    processor = TrOCRProcessor.from_pretrained(ModelConfig.MODEL_NAME)

    # Create datasets for training and validation
    train_dataset = HandleDataTrOCR(hdf5_file_path, size_train, processor)
    valid_dataset = HandleDataTrOCR(hdf5_file_path, 'valid', processor)

    # Load the pre-trained model
    model = VisionEncoderDecoderModel.from_pretrained(ModelConfig.MODEL_NAME)
    model.to(device)

    # Model configuration (same as before)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.bos_token_id = processor.tokenizer.cls_token_id

    # Beam search parameters (same as before)
    model.config.max_length = 32
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 8

    # CER computation function (same as before)
    def compute_cer(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
        cer = metrics_evaluation.cer_only([pred_str], [label_str])
        return {"cer": cer}

    # Training arguments (same as before)
    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy='epoch',
        per_device_train_batch_size=training_config.BATCH_SIZE,
        per_device_eval_batch_size=training_config.BATCH_SIZE,
        output_dir=model_save_dir,
        logging_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=5,
        learning_rate=training_config.LEARNING_RATE,
        report_to='tensorboard',
        num_train_epochs=training_config.EPOCHS,
        fp16=True
    )

    # Trainer setup (same as before)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_cer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=default_data_collator
    )

    # Train the model (same as before)
    res = trainer.train()
    print(res)

    # Save model and processor (same as before)
    os.makedirs(model_save_dir, exist_ok=True)
    model.save_pretrained(model_save_dir)
    processor.save_pretrained(processor_save_dir)
    print(f"Model saved to {model_save_dir}")
    print(f"Processor saved to {processor_save_dir}")

    # Clean up resources
    train_dataset.close()  # Close HDF5 file for train dataset
    valid_dataset.close()  # Close HDF5 file for validation dataset
    clear_cuda_cache()