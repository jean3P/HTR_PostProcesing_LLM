import datetime
import os

from config.flor_parameters import HTRFlorConfig
from utils.flor.data.generator import DataGenerator
from utils.flor.network.model import HTRModel


def train_model(dtgen, model_output_path, checkpoitn_path):
    model = HTRModel(
        architecture="flor",
        input_size=HTRFlorConfig.INPUT_SIZE,
        vocab_size=dtgen.tokenizer.vocab_size,
        beam_width=HTRFlorConfig.BEAM_WIDTH,
        stop_tolerance=HTRFlorConfig.STOP_TOLERANCE,
        reduce_tolerance=HTRFlorConfig.REDUCE_TOLERANCE,
        reduce_factor=HTRFlorConfig.REDUCE_FACTOR
    )
    model.compile(learning_rate=HTRFlorConfig.LEARNING_RATE)
    model.summary(model_output_path, "summary.txt")
    model.load_checkpoint(target=checkpoitn_path)

    callbacks = model.get_callbacks(logdir=model_output_path, checkpoint=checkpoitn_path, verbose=1)
    start_time = datetime.datetime.now()

    history = model.fit(
        x=dtgen.next_train_batch(),
        epochs=HTRFlorConfig.EPOCHS,
        steps_per_epoch=dtgen.steps[dtgen.train_partition],
        validation_data=dtgen.next_valid_batch(),
        validation_steps=dtgen.steps['valid'],
        callbacks=callbacks,
        shuffle=True,
        verbose=1
    )

    total_time = datetime.datetime.now() - start_time
    summarize_training(history, total_time, dtgen, model_output_path)
    return model


def get_data_generator(hdf5_file_path, train_partition):
    """ Setup the data generator for the project. """
    dtgen = DataGenerator(
        source=hdf5_file_path,
        batch_size=HTRFlorConfig.BATCH_SIZE,
        charset=HTRFlorConfig.CHARSET_BASE,
        max_text_length=HTRFlorConfig.MAX_TEXT_LENGTH,
        train_partition=train_partition
    )

    print(f"Train images: {dtgen.size[train_partition]}")
    print(f"Validation images: {dtgen.size['valid']}")
    print(f"Test images: {dtgen.size['test']}")
    return dtgen


def summarize_training(history, total_time, dtgen, output_path):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    min_val_loss = min(val_loss)
    min_val_loss_i = val_loss.index(min_val_loss)
    time_epoch = (total_time / len(loss))
    total_item = (dtgen.size[dtgen.train_partition] + dtgen.size['valid'])

    summary = f"""
    Total train images:      {dtgen.size[dtgen.train_partition]}
    Total validation images: {dtgen.size['valid']}
    Batch:                   {dtgen.batch_size}
    Total time:              {total_time}
    Time per epoch:          {time_epoch}
    Time per item:           {time_epoch / total_item}
    Total epochs:            {len(loss)}
    Best epoch               {min_val_loss_i + 1}
    Training loss:           {loss[min_val_loss_i]:.8f}
    Validation loss:         {min_val_loss:.8f}
    """

    with open(os.path.join(output_path, "train.txt"), "w") as file:
        file.write(summary)
    print(summary)
