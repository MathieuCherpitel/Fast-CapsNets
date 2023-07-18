from tqdm import tqdm
import json

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


class CapsNet(tf.keras.Model):
    def __init__(self, epochs, epsilon, m_minus, m_plus, lambda_, alpha, no_of_conv_kernels, no_of_primary_capsules, primary_capsule_vector, no_of_secondary_capsules, secondary_capsule_vector, r):
        super(CapsNet, self).__init__()
        self.epochs = epochs
        self.epsilon = epsilon
        self.m_minus = m_minus
        self.m_plus = m_plus
        self.lambda_ = lambda_
        self.alpha = alpha
        self.no_of_conv_kernels = no_of_conv_kernels
        self.no_of_primary_capsules = no_of_primary_capsules
        self.primary_capsule_vector = primary_capsule_vector
        self.no_of_secondary_capsules = no_of_secondary_capsules
        self.secondary_capsule_vector = secondary_capsule_vector
        self.r = r
        self.training_metrics = None
        self.reconstruct = True

        self.early_stop = True # in case the validation loss decreases, stops the training if True

        with tf.name_scope("Variables") as scope:
            self.convolution = tf.keras.layers.Conv2D(self.no_of_conv_kernels, [9,9], strides=[1,1], name='ConvolutionLayer', activation='relu')
            self.primary_capsule = tf.keras.layers.Conv2D(self.no_of_primary_capsules * self.primary_capsule_vector, [9,9], strides=[2,2], name="PrimaryCapsule")
            self.w = tf.Variable(tf.random_normal_initializer()(shape=[1, 1152, self.no_of_secondary_capsules, self.secondary_capsule_vector, self.primary_capsule_vector]), dtype=tf.float32, name="PoseEstimation", trainable=True)
            self.dense_1 = tf.keras.layers.Dense(units = 512, activation='relu')
            self.dense_2 = tf.keras.layers.Dense(units = 1024, activation='relu')
            self.dense_3 = None

        self.build(input_shape=())

    def get_config(self):
        return {
            "epochs": self.epochs,
            "epsilon": self.epsilon,
            "m_minus": self.m_minus,
            "m_plus": self.m_plus,
            "lambda_": self.lambda_,
            "alpha": self.alpha,
            "no_of_conv_kernels": self.no_of_conv_kernels,
            "no_of_primary_capsules": self.no_of_primary_capsules,
            "no_of_secondary_capsules": self.no_of_secondary_capsules,
            "primary_capsule_vector": self.primary_capsule_vector,
            "secondary_capsule_vector": self.secondary_capsule_vector,
            "r": self.r,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def squash(self, s):
        with tf.name_scope("SquashFunction") as scope:
            s_norm = tf.norm(s, axis=-1, keepdims=True)
            return tf.square(s_norm)/(1 + tf.square(s_norm)) * s/(s_norm + self.epsilon)

    @tf.function
    def call(self, inputs):
        input_x, y = inputs
        # input_x.shape: (None, 28, 28, 1)
        # y.shape: (None, 10)

        x = self.convolution(input_x) # x.shape: (None, 20, 20, 256)
        x = self.primary_capsule(x) # x.shape: (None, 6, 6, 256)

        with tf.name_scope("CapsuleFormation") as scope:
            u = tf.reshape(x, (-1, self.no_of_primary_capsules * x.shape[1] * x.shape[2], 8)) # u.shape: (None, 1152, 8)
            u = tf.expand_dims(u, axis=-2) # u.shape: (None, 1152, 1, 8)
            u = tf.expand_dims(u, axis=-1) # u.shape: (None, 1152, 1, 8, 1)
            u_hat = tf.matmul(self.w, u) # u_hat.shape: (None, 1152, 10, 16, 1)
            u_hat = tf.squeeze(u_hat, [4]) # u_hat.shape: (None, 1152, 10, 16)

        with tf.name_scope("DynamicRouting") as scope:
            b = tf.zeros((input_x.shape[0] if input_x.shape[0] else 1, 1152, self.no_of_secondary_capsules, 1)) # b.shape: (None, 1152, 10, 1)
            for i in range(self.r): # self.r = 3
                c = tf.nn.softmax(b, axis=-2) # c.shape: (None, 1152, 10, 1)
                s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True) # s.shape: (None, 1, 10, 16)
                v = self.squash(s) # v.shape: (None, 1, 10, 16)
                agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4]) # agreement.shape: (None, 1152, 10, 1)
                # Before matmul following intermediate shapes are present, they are not assigned to a variable but just for understanding the code.
                # u_hat.shape (Intermediate shape) : (None, 1152, 10, 16, 1)
                # v.shape (Intermediate shape): (None, 1, 10, 16, 1)
                # Since the first parameter of matmul is to be transposed its shape becomes:(None, 1152, 10, 1, 16)
                # Now matmul is performed in the last two dimensions, and others are broadcasted
                # Before squeezing we have an intermediate shape of (None, 1152, 10, 1, 1)
                b += agreement

        with tf.name_scope("Masking") as scope:
            y = tf.expand_dims(y, axis=-1) # y.shape: (None, 10, 1)
            y = tf.expand_dims(y, axis=1) # y.shape: (None, 1, 10, 1)
            mask = tf.cast(y, dtype=tf.float32) # mask.shape: (None, 1, 10, 1)
            v_masked = tf.multiply(mask, v) # v_masked.shape: (None, 1, 10, 16)

        with tf.name_scope("Reconstruction") as scope:
            v_ = tf.reshape(v_masked, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector]) # v_.shape: (None, 160)
            reconstructed_image = self.dense_1(v_) # reconstructed_image.shape: (None, 512)
            reconstructed_image = self.dense_2(reconstructed_image) # reconstructed_image.shape: (None, 1024)
            if not self.dense_3:
                self.dense_3 = tf.keras.layers.Dense(units = input_x.shape[1] * input_x.shape[2], activation='sigmoid', dtype='float32')
            reconstructed_image = self.dense_3(reconstructed_image) # reconstructed_image.shape: (None, image_size)
        return v, reconstructed_image

    @tf.function
    def predict_capsule_output(self, inputs):
        x = self.convolution(inputs) # x.shape: (None, 20, 20, 256)
        x = self.primary_capsule(x) # x.shape: (None, 6, 6, 256)
        
        with tf.name_scope("CapsuleFormation") as scope:
            u = tf.reshape(x, (-1, self.no_of_primary_capsules * x.shape[1] * x.shape[2], 8)) # u.shape: (None, 1152, 8)
            u = tf.expand_dims(u, axis=-2) # u.shape: (None, 1152, 1, 8)
            u = tf.expand_dims(u, axis=-1) # u.shape: (None, 1152, 1, 8, 1)
            u_hat = tf.matmul(self.w, u) # u_hat.shape: (None, 1152, 10, 16, 1)
            u_hat = tf.squeeze(u_hat, [4]) # u_hat.shape: (None, 1152, 10, 16)

        
        with tf.name_scope("DynamicRouting") as scope:
            b = tf.zeros((inputs.shape[0], 1152, self.no_of_secondary_capsules, 1)) # b.shape: (None, 1152, 10, 1)
            for i in range(self.r): # self.r = 3
                c = tf.nn.softmax(b, axis=-2) # c.shape: (None, 1152, 10, 1)
                s = tf.reduce_sum(tf.multiply(c, u_hat), axis=1, keepdims=True) # s.shape: (None, 1, 10, 16)
                v = self.squash(s) # v.shape: (None, 1, 10, 16)
                agreement = tf.squeeze(tf.matmul(tf.expand_dims(u_hat, axis=-1), tf.expand_dims(v, axis=-1), transpose_a=True), [4]) # agreement.shape: (None, 1152, 10, 1)
                # Before matmul following intermediate shapes are present, they are not assigned to a variable but just for understanding the code.
                # u_hat.shape (Intermediate shape) : (None, 1152, 10, 16, 1)
                # v.shape (Intermediate shape): (None, 1, 10, 16, 1)
                # Since the first parameter of matmul is to be transposed its shape becomes:(None, 1152, 10, 1, 16)
                # Now matmul is performed in the last two dimensions, and others are broadcasted
                # Before squeezing we have an intermediate shape of (None, 1152, 10, 1, 1)
                b += agreement
        return v

    @tf.function
    def regenerate_image(self, inputs):
        with tf.name_scope("Reconstruction") as scope:
            v_ = tf.reshape(inputs, [-1, self.no_of_secondary_capsules * self.secondary_capsule_vector]) # v_.shape: (None, 160)
            reconstructed_image = self.dense_1(v_) # reconstructed_image.shape: (None, 512)
            reconstructed_image = self.dense_2(reconstructed_image) # reconstructed_image.shape: (None, 1024)
            reconstructed_image = self.dense_3(reconstructed_image) # reconstructed_image.shape: (None, image_size)
        return reconstructed_image

    def safe_norm(self, v, axis=-1):
        v_ = tf.reduce_sum(tf.square(v), axis = axis, keepdims=True)
        return tf.sqrt(v_ + self.epsilon)

    def loss_function(self, dataset, depth, optimizer, X, pbar, val=None):
        final_loss = 0
        for X_batch, y_batch in dataset:
            y_one_hot = tf.one_hot(y_batch, depth=depth)
            with tf.GradientTape() as tape:
                v, reconstructed_image = self([X_batch, y_one_hot])
                prediction = self.safe_norm(v)
                prediction = tf.reshape(prediction, [-1, self.no_of_secondary_capsules])
                left_margin = tf.square(tf.maximum(0.0, self.m_plus - prediction))
                right_margin = tf.square(tf.maximum(0.0, prediction - self.m_minus))
                l = tf.add(y_one_hot * left_margin, self.lambda_ * (1.0 - y_one_hot) * right_margin)
                margin_loss = tf.reduce_mean(tf.reduce_sum(l, axis=-1))
                loss = margin_loss
                if X_batch.shape[-1] == 1:
                    # For one channel images, compute the reconstruction loss
                    y_image_flat = tf.reshape(X_batch, [-1, X_batch.shape[1] * X_batch.shape[2]])
                    reconstruction_loss = tf.reduce_mean(tf.square(y_image_flat - reconstructed_image))
                    loss = tf.add(loss, self.alpha * reconstruction_loss)
                final_loss += loss
            if not val:
                grad = tape.gradient(loss, self.trainable_variables)
                optimizer.apply_gradients(zip(grad, self.trainable_variables))
                pbar.update(1)
        return final_loss / len(X)

    def compute_train_metrics(self, train_metrics, metrics, X, y,  validation):
        y_preds = self.predict(X)
        if 'accuracy' in train_metrics:
            metrics['accuracy'].append(accuracy_score(y, y_preds))
        if 'f1' in train_metrics:
            metrics['f1'].append(f1_score(y, y_preds, average='weighted'))
        if 'precision' in train_metrics:
            metrics['precision'].append(precision_score(y, y_preds, average='weighted'))
        if 'recall' in train_metrics:
            metrics['recall'].append(recall_score(y, y_preds, average='weighted'))
        if validation:
            val_X = validation[0]
            val_y = validation[1]
            val_y_preds = self.predict(val_X)
            if 'accuracy' in train_metrics:
                metrics['val_accuracy'].append(accuracy_score(val_y, val_y_preds))
            if 'f1' in train_metrics:
                metrics['val_f1'].append(f1_score(val_y, val_y_preds, average='weighted'))
            if 'precision' in train_metrics:
                metrics['val_precision'].append(precision_score(val_y, val_y_preds, average='weighted'))
            if 'recall' in train_metrics:
                metrics['val_recall'].append(recall_score(val_y, val_y_preds, average='weighted'))
        return metrics

    # During training, the model is evaluated on a holdout validation dataset after each epoch.
    # If the performance of the model on the validation dataset starts to degrade (e.g. loss begins to increase or accuracy begins to decrease),
    # then the training process is stopped.
    def fit(self, X, y, optimizer, batch_size=64, train_metrics=None, validation=None):
        metrics = dict((el,[]) for el in train_metrics) if train_metrics else {}
        metrics['loss'] = []
        training = tf.data.Dataset.from_tensor_slices((X, y))
        training = training.shuffle(buffer_size=len(training), reshuffle_each_iteration=True)
        training = training.batch(batch_size=batch_size)
        if validation:
            metrics.update(dict((f"val_{el}",[]) for el in train_metrics) if train_metrics else {})
            metrics['val_loss'] = []
            validating = tf.data.Dataset.from_tensor_slices((validation[0], validation[1]))
            validating = validating.shuffle(buffer_size=len(validating), reshuffle_each_iteration=True)
            validating = validating.batch(batch_size=batch_size)
        depth = len(np.unique(y))
        train_loss = 0
        val_loss = 1
        for i in range(1, self.epochs+1, 1):
            with tqdm(total=len(training)) as pbar:
                pbar.set_description_str(f"Epoch {i}/{self.epochs}")
                train_loss = self.loss_function(training, depth, optimizer, X, pbar, val=None)
                if validation:
                    old = val_loss
                    val_loss = self.loss_function(validating, depth, optimizer, validation[0], pbar, val=True)
                    # early stopping
                    if old < val_loss and self.early_stop:
                        print(f"Validation loss is not decreasing anymore, risk of overfitting the model after {i} epochs. The training will be stopped. This behavior can be modified by setting the early_stop property of the model to False")
                        self.epochs = i # set the real number of epochs required to train the model before overfitting.
                        self.training_metrics = metrics
                        return metrics
                if train_metrics:
                    pbar.set_postfix_str("Evaluating ...")
                    metrics = self.compute_train_metrics(train_metrics, metrics, X, y, validation)
                metrics['loss'].append(float(train_loss.numpy()))
                if validation:
                    metrics['val_loss'].append(float(val_loss.numpy()))
                    pbar.set_postfix_str(f"Training loss : {train_loss.numpy():.4f}, Validation loss : {val_loss.numpy():.4f}")
                else :
                    pbar.set_postfix_str(f"Loss : {train_loss.numpy():.4f}")
        self.training_metrics = metrics
        return metrics

    def predict_proba(self, X):
        # In case X is a single element, create a batch of size 1
        if X.ndim == 3:
            X = tf.expand_dims(X, axis=0)
        pred = self.safe_norm(self.predict_capsule_output(X))
        pred = tf.squeeze(pred, [1])
        return pred

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)[:,0]

    def save(self):
        # Calling tensorflow save method to save the model
        print(f"Saving model under /saved_models/{self.name}")
        super(CapsNet, self).save(f"../saved_models/{self.name}")
        if not self.training_metrics:
            raise ValueError('The model bust be trained before being saved')
        # Model config
        with open(f'../saved_models/{self.name}/assets/config.json', 'w') as fp:
            json.dump(self.get_config(), fp)
        # Training metrics
        df = pd.DataFrame(self.training_metrics)
        plot = df.plot(title=f"Metrics {self.name}")
        plot.set_ylim(0, 1)
        plot.set_xlim(left=0)
        plot.set_xlabel("epochs")
        plot.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        with open(f'../saved_models/{self.name}/assets/training_metrics.json', 'w') as fp:
            json.dump(self.training_metrics, fp)
        plot.figure.savefig(f'../saved_models/{self.name}/assets/training_metrics.pdf')
        self.saved = True

    def evaluate(self, X_test, y_test, classes):
        preds = self.predict(X_test)
        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        df_cm = pd.DataFrame(cm, index=classes, columns=classes)
        sn.heatmap(df_cm, annot=True)
        # Classification metrics
        report = classification_report(y_test, preds, output_dict=True)
        df = pd.DataFrame(report).transpose()
        df.index = classes + ['accuracy', 'macro avg', 'weighted avg']
        return df
