from __future__ import annotations

import tensorflow as tf


def main() -> None:
    print("tensorflow", tf.__version__)
    print("built_cuda", tf.test.is_built_with_cuda())
    print("physical_gpus", tf.config.list_physical_devices("GPU"))


if __name__ == "__main__":
    main()
