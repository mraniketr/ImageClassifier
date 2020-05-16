{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from os import path, getcwd, chdir\n",
    "\n",
    "path = f\"{getcwd()}/../tmp2/mnist.npz\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "config = tf.ConfigProto()\n",
    "config.gpu_options.allow_growth = True\n",
    "sess = tf.Session(config=config)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train_mnist_conv():\n",
    "    class myCallback(tf.keras.callbacks.Callback):\n",
    "        def on_epoch_end(self, epoch, logs={}):\n",
    "            if(logs.get('acc')>0.998):\n",
    "                print(\"\\nReached 99.8% accuracy so cancelling training!\")\n",
    "                self.model.stop_training = True\n",
    "\n",
    "    mnist = tf.keras.datasets.mnist\n",
    "    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)\n",
    "    training_images=training_images.reshape(60000, 28, 28, 1)\n",
    "    training_images=training_images / 255.0\n",
    "    test_images = test_images.reshape(10000, 28, 28, 1)\n",
    "    test_images=test_images/255.0\n",
    "\n",
    "    callbacks=myCallback()\n",
    "\n",
    "    model = tf.keras.models.Sequential([\n",
    "            tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),\n",
    "            tf.keras.layers.MaxPooling2D(2, 2),\n",
    "            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),\n",
    "            tf.keras.layers.MaxPooling2D(2,2),\n",
    "            tf.keras.layers.Flatten(), \n",
    "            tf.keras.layers.Dense(128, activation=tf.nn.relu), \n",
    "            tf.keras.layers.Dense(10, activation=tf.nn.softmax)\n",
    "    ])\n",
    "\n",
    "    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
    "    model.summary()\n",
    "\n",
    "    history = model.fit(\n",
    "        training_images,\n",
    "        training_labels,\n",
    "        epochs=15,\n",
    "        callbacks=[callbacks]\n",
    "    )\n",
    "    return history.epoch, history.history['acc'][-1]\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_1\"\n",
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_2 (Conv2D)            (None, 26, 26, 64)        640       \n",
      "_________________________________________________________________\n",
      "max_pooling2d_2 (MaxPooling2 (None, 13, 13, 64)        0         \n",
      "_________________________________________________________________\n",
      "conv2d_3 (Conv2D)            (None, 11, 11, 64)        36928     \n",
      "_________________________________________________________________\n",
      "max_pooling2d_3 (MaxPooling2 (None, 5, 5, 64)          0         \n",
      "_________________________________________________________________\n",
      "flatten_1 (Flatten)          (None, 1600)              0         \n",
      "_________________________________________________________________\n",
      "dense_2 (Dense)              (None, 128)               204928    \n",
      "_________________________________________________________________\n",
      "dense_3 (Dense)              (None, 10)                1290      \n",
      "=================================================================\n",
      "Total params: 243,786\n",
      "Trainable params: 243,786\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n",
      "Epoch 1/15\n",
      "60000/60000 [==============================] - 18s 295us/sample - loss: 0.1226 - acc: 0.9618\n",
      "Epoch 2/15\n",
      "60000/60000 [==============================] - 18s 297us/sample - loss: 0.0414 - acc: 0.9870\n",
      "Epoch 3/15\n",
      "60000/60000 [==============================] - 17s 290us/sample - loss: 0.0285 - acc: 0.9911\n",
      "Epoch 4/15\n",
      "60000/60000 [==============================] - 17s 291us/sample - loss: 0.0203 - acc: 0.9940\n",
      "Epoch 5/15\n",
      "60000/60000 [==============================] - 17s 285us/sample - loss: 0.0152 - acc: 0.9950\n",
      "Epoch 6/15\n",
      "60000/60000 [==============================] - 17s 284us/sample - loss: 0.0114 - acc: 0.9965\n",
      "Epoch 7/15\n",
      "60000/60000 [==============================] - 18s 293us/sample - loss: 0.0104 - acc: 0.9965\n",
      "Epoch 8/15\n",
      "60000/60000 [==============================] - 18s 294us/sample - loss: 0.0076 - acc: 0.9977\n",
      "Epoch 9/15\n",
      "59936/60000 [============================>.] - ETA: 0s - loss: 0.0064 - acc: 0.9981\n",
      "Reached 99.8% accuracy so cancelling training!\n",
      "60000/60000 [==============================] - 18s 308us/sample - loss: 0.0064 - acc: 0.9980\n"
     ]
    }
   ],
   "source": [
    "_, _ = train_mnist_conv()"
   ]
  }
 ],
 "metadata": {
  "coursera": {
   "course_slug": "introduction-tensorflow",
   "graded_item_id": "ml06H",
   "launcher_item_id": "hQF8A"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
