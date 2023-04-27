# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import functools
import logging
import os
import sys

from termcolor import colored


def plot_image(
    writer, data, position="Train/input", step=0, norm_fn=None
):  # pylint: disable=missing-docstring
    """
    The function plots an image using a given writer, data, position, step, and normalization function.

    Args:
      writer: This is an instance of a SummaryWriter object from the PyTorch library. It is used to
    write data to a TensorBoard log file.
      data: The input data to be plotted as an image. It can be a tensor or any object that has a method
    called "as_tensor" to convert it to a tensor.
      position: The position parameter specifies the name of the image plot in the TensorBoard. It is
    used to organize the plots in the TensorBoard dashboard. In this function, the default value of
    position is "Train/input", which means that the plot will be displayed under the "Train" tab and
    labeled as ". Defaults to Train/input
      step: The step parameter is an integer that represents the current step or iteration of the
    training process. It is used to keep track of the progress of the model during training and is often
    used in conjunction with a visualization tool like TensorBoard to plot the performance of the model
    over time. Defaults to 0
      norm_fn: norm_fn is a normalization function that can be applied to the input data before
    plotting. It is an optional parameter and can be set to None if no normalization is required.

    Returns:
      the input data after normalizing it (if a normalization function is provided) and displaying it as
    images using the `add_images` method of the `writer` object. The images are added to the `position`
    specified (default is "Train/input") and the `step` specified (default is 0). The function returns
    the input data as a tensor.
    """
    data = data.as_tensor() if hasattr(data, "as_tensor") else data
    if norm_fn is not None:
        data = norm_fn(data)
    batch_size = data.size()[0]
    writer.add_images(position, data[batch_size // 4 :], step)
    return data


def plot_line(
    writer, data, position="Train/input", step=0
):  # pylint: disable=missing-docstring
    """
    This function adds a scalar value to a tensor and plots it on a graph using a specified position and
    step.

    Args:
      writer: The writer object is an instance of a SummaryWriter class from the PyTorch library. It is
    used to write scalar values, images, histograms, and other data to a TensorBoard log file.
      data: The data to be plotted on the graph. It can be a tensor or any other data type that can be
    converted to a tensor using the `as_tensor()` method.
      position: The position parameter specifies the name of the plot or graph that will be displayed in
    the TensorBoard. It is used to organize and group different plots or graphs together. In this case,
    the default value for position is "Train/input", which suggests that this plot is related to the
    input data during training. Defaults to Train/input
      step: The step parameter is an integer value that represents the current step or iteration number
    of the training process. It is used to keep track of the progress of the training and to plot the
    data at different stages of the training process. Defaults to 0

    Returns:
      the `data` argument that was passed to it.
    """
    data = data.as_tensor() if hasattr(data, "as_tensor") else data
    writer.add_scalar(position, data, step)
    return data


@functools.lru_cache()
def create_logger(
    output_dir, dist_rank=0, name=""
):  # pylint: disable=missing-docstring
    """
    This function creates a logger object with console and file handlers for logging messages.

    Args:
      output_dir: The directory where the log files will be saved.
      dist_rank: The dist_rank parameter is used to specify the rank of the distributed process. It is
    used to determine whether to create console handlers for the master process or not. If dist_rank is
    0, console handlers will be created for the master process. Defaults to 0
      name: The name of the logger. It is an optional parameter and if not provided, the root logger is
    used.

    Returns:
      a logger object that has been configured with console and file handlers for logging messages. The
    logger object can be used to log messages at different levels of severity (e.g. debug, info,
    warning, error, critical) and these messages will be output to the console and/or file depending on
    the configuration. The logger object can be customized with a specific name and output directory.
    The
    """
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = "[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s"
    color_fmt = (
        colored("[%(asctime)s %(name)s]", "green")
        + colored("(%(filename)s %(lineno)d)", "yellow")
        + ": %(levelname)s %(message)s"
    )

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            # logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
            logging.Formatter(fmt=color_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(console_handler)

    # create file handlers
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(output_dir, f"log_rank{dist_rank}.txt"), mode="a"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    return logger
