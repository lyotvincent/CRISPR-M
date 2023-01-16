import tensorflow.keras as keras
from sklearn.metrics import roc_auc_score, average_precision_score
import tensorflow as tf

class AUROCMetric(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_state(self, *args, **kwargs):
        print("AUROCMetric update_state")

    def result(self):
        print("AUROCMetric result")
        return 0

    def get_config(self):
        return super().get_config()

class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = self.create_huber(threshold)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        print("AUROCMetric update_state")
        print("AUROCMetric update_state y_true shape = %s"%str(y_true.shape))
        print(y_true)
        print("AUROCMetric update_state y_pred shape = %s"%str(y_pred.shape))
        print(y_pred)
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        print("AUROCMetric result")
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}

    def create_huber(self, threshold=1.0):
        def huber_fn(y_true, y_pred):
            error = y_true - y_pred
            is_small_error = tf.abs(error) < threshold
            squared_loss = tf.square(error) / 2
            linear_loss = threshold * tf.abs(error) - threshold**2 / 2
            return tf.where(is_small_error, squared_loss, linear_loss)
        return huber_fn
