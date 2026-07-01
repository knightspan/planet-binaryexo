"""
pipeline/classifier.py
=======================
Dual-view 1D CNN classifier — architecture from Shallue & Vanderburg (2018)
AstroNet, extended with deeper convolutions and a physics-informed rule
ensemble.

TWO INPUTS:
  Global view (2001 points, phase −0.5 to +0.5):
    Captures full orbital context, secondary eclipses, overall light-curve
    shape. Allows the model to see the transit in context.

  Local view (201 points, phase ±0.075):
    Zoomed on the transit itself. Captures ingress/egress curvature, flat
    vs V-shaped bottom — the key discriminator between planets and EBs.

TWO BRANCHES merged → dense head → 4-class softmax:
  0: noise          1: planet_transit
  2: eclipsing_binary  3: blend

RULE-BASED ENSEMBLE:
  Physics-informed soft classifier running in parallel. Weighted 60/40
  with CNN when CNN weights are available; 100% rules otherwise.
  Interpretable — every decision can be traced back to a physical test.
"""

import numpy as np
import logging
import os

log = logging.getLogger(__name__)

CLASSES    = ['noise', 'planet_transit', 'eclipsing_binary', 'blend']
N_GLOBAL   = 2001
N_LOCAL    = 201


class ExoplanetClassifier:

    def __init__(self, config: dict):
        self.config     = config
        self.model_path = config.get('model_path', 'models/classifier.h5')
        self.model      = None
        self._loaded    = False

    # ─────────────────────────────────────────────────────────────────────────
    # PREDICT
    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, time, flux, tls_result, vetting_flags: dict):
        """Returns (label_str, confidence_float, probs_array[4])."""
        gv, lv   = self._build_views(time, flux, tls_result)
        feat_vec = self._build_features(tls_result, vetting_flags)

        rule_probs = self._rule_based(tls_result, vetting_flags)

        if self._try_load_model():
            cnn_probs  = self._cnn_predict(gv, lv)
            probs      = 0.60 * cnn_probs + 0.40 * rule_probs
        else:
            probs = rule_probs

        probs      = np.clip(probs, 1e-6, 1.0)
        probs     /= probs.sum()
        idx        = int(np.argmax(probs))
        return CLASSES[idx], float(probs[idx]), probs

    # ─────────────────────────────────────────────────────────────────────────
    # VIEWS
    # ─────────────────────────────────────────────────────────────────────────
    def _build_views(self, time, flux, r):
        from pipeline.preprocessor import LightCurvePreprocessor
        _, _, gv, lv = LightCurvePreprocessor.phase_fold(
            time, flux, r.period, r.T0,
            n_bins_global=N_GLOBAL, n_bins_local=N_LOCAL)
        gv = _minmax(gv)
        lv = _minmax(lv)
        return gv, lv

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE VECTOR (14-d) for RF / interpretability
    # ─────────────────────────────────────────────────────────────────────────
    def _build_features(self, r, flags):
        return np.array([
            float(r.period),
            float(r.depth),
            float(r.duration),
            float(r.SDE),
            float(r.transit_count or 0),
            float(r.odd_even_mismatch or 1.0),
            float(flags.get('odd_even_ratio',    1.0)),
            float(flags.get('secondary_depth',   0.0)),
            float(flags.get('secondary_snr',     0.0)),
            float(flags.get('centroid_shift',    0.0)),
            float(flags.get('depth_mean',        r.depth)),
            float(flags.get('depth_std',         0.0)),
            float(np.log10(max(r.period,  0.001))),
            float(np.log10(max(r.depth,   1e-7))),
        ], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # PHYSICS-INFORMED RULE-BASED CLASSIFIER
    # ─────────────────────────────────────────────────────────────────────────
    def _rule_based(self, r, flags):
        depth    = float(r.depth)
        sde      = float(r.SDE)
        n_trans  = int(r.transit_count or 0)
        oe       = float(flags.get('odd_even_ratio',    1.0))
        sec_d    = float(flags.get('secondary_depth',   0.0))
        sec_snr  = float(flags.get('secondary_snr',     0.0))
        centroid = float(flags.get('centroid_shift',    0.0))
        v_shape  = float(flags.get('v_shape_metric',    0.8))
        fp_score = int(flags.get('fp_indicators',       0))

        p = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)

        # ── noise evidence ────────────────────────────────────────────────────
        if sde < 7.0:       p[0] += 0.5
        if n_trans < 3:     p[0] += 0.2
        if depth < 2e-4:    p[0] += 0.2   # < 200 ppm

        # ── EB evidence ───────────────────────────────────────────────────────
        if depth > 0.01:              p[2] += 0.35  # > 1% deep
        if depth > 0.05:              p[2] += 0.40  # > 5% very deep
        if abs(oe - 1.0) > 0.15:     p[2] += 0.35  # odd/even mismatch
        if sec_snr > 3.0:             p[2] += 0.30  # secondary eclipse seen
        if v_shape < 0.4:             p[2] += 0.25  # V-shaped

        # ── blend evidence ────────────────────────────────────────────────────
        if centroid > 0.5:            p[3] += 0.50  # centroid shifts
        if 5e-4 < depth < 0.008 and centroid > 0.3:
                                      p[3] += 0.25

        bc = flags.get('blend_crosscheck', {})
        if isinstance(bc, dict) and bc.get('blend_likely', False):
            p[3] += 0.35  # catalog-based blend evidence

        # ── planet transit evidence ───────────────────────────────────────────
        if sde >= 9.0:                p[1] += 0.30
        if 5e-4 < depth < 0.02:      p[1] += 0.30  # realistic planet depth
        if n_trans >= 3:              p[1] += 0.20
        if abs(oe - 1.0) < 0.10:     p[1] += 0.20  # consistent odd/even
        if sec_snr < 2.0:             p[1] += 0.10  # no secondary
        if centroid < 0.3:            p[1] += 0.10  # stable centroid
        if v_shape > 0.6:             p[1] += 0.10  # flat bottom
        if fp_score == 0:             p[1] += 0.15  # all vetting passed

        p = np.clip(p, 0.01, None)
        return p / p.sum()

    # ─────────────────────────────────────────────────────────────────────────
    # CNN
    # ─────────────────────────────────────────────────────────────────────────
    def build_model(self):
        """Build dual-view 1D CNN. Call this before training."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            K = keras.layers
        except ImportError:
            log.error("TensorFlow not installed. Run: pip install tensorflow")
            return None

        # Global branch
        g_in = keras.Input(shape=(N_GLOBAL, 1), name='global_view')
        g    = K.Conv1D(16, 5, activation='relu', padding='same')(g_in)
        g    = K.BatchNormalization()(g)
        g    = K.MaxPooling1D(4)(g)
        g    = K.Conv1D(32, 5, activation='relu', padding='same')(g)
        g    = K.BatchNormalization()(g)
        g    = K.MaxPooling1D(4)(g)
        g    = K.Conv1D(64, 3, activation='relu', padding='same')(g)
        g    = K.MaxPooling1D(4)(g)
        g    = K.Conv1D(128, 3, activation='relu', padding='same')(g)
        g    = K.GlobalAveragePooling1D()(g)

        # Local branch
        l_in = keras.Input(shape=(N_LOCAL, 1), name='local_view')
        l    = K.Conv1D(16, 3, activation='relu', padding='same')(l_in)
        l    = K.BatchNormalization()(l)
        l    = K.MaxPooling1D(2)(l)
        l    = K.Conv1D(32, 3, activation='relu', padding='same')(l)
        l    = K.BatchNormalization()(l)
        l    = K.MaxPooling1D(2)(l)
        l    = K.Conv1D(64, 3, activation='relu', padding='same')(l)
        l    = K.GlobalAveragePooling1D()(l)

        # Merge + classify
        merged = K.Concatenate()([g, l])
        x      = K.Dense(256, activation='relu')(merged)
        x      = K.Dropout(0.35)(x)
        x      = K.Dense(128, activation='relu')(x)
        x      = K.Dropout(0.35)(x)
        x      = K.Dense(64,  activation='relu')(x)
        out    = K.Dense(4, activation='softmax', name='class_output')(x)

        model  = keras.Model(inputs=[g_in, l_in], outputs=out,
                              name='JyotirVega_DualViewCNN')
        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=1e-4),
            loss      = 'sparse_categorical_crossentropy',
            metrics   = ['accuracy']
        )
        log.info(f"  DualViewCNN built: {model.count_params():,} parameters")
        return model

    def _try_load_model(self):
        if self._loaded:
            return self.model is not None
        self._loaded = True
        if os.path.exists(self.model_path):
            try:
                from tensorflow import keras
                self.model = keras.models.load_model(self.model_path)
                log.info(f"  Loaded CNN weights from {self.model_path}")
                return True
            except Exception as e:
                log.warning(f"  Could not load CNN: {e}")
        return False

    def _cnn_predict(self, global_view, local_view):
        g = global_view.reshape(1, N_GLOBAL, 1)
        l = local_view.reshape(1,  N_LOCAL,  1)
        probs = self.model.predict(
            {'global_view': g, 'local_view': l}, verbose=0)[0]
        return np.array(probs, dtype=np.float64)

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────────────────────────────────
    def train(self, X_global, X_local, X_features, y_labels,
              epochs=50, batch_size=32, val_split=0.2, save_path=None):
        """
        Train the dual-view CNN.

        X_global   : (N, 2001) float32
        X_local    : (N, 201)  float32
        X_features : (N, 14)   float32  (not used by CNN, kept for RF)
        y_labels   : (N,)      int32    class indices 0-3
        """
        try:
            from tensorflow import keras
        except ImportError:
            log.error("TensorFlow not installed.")
            return None

        model = self.build_model()
        if model is None:
            return None

        Xg = X_global.reshape(-1, N_GLOBAL, 1)
        Xl = X_local.reshape(-1,  N_LOCAL,  1)

        # Class weights to handle imbalance
        from collections import Counter
        counts  = Counter(y_labels.tolist())
        total   = len(y_labels)
        cw      = {i: total / (4 * counts.get(i, 1)) for i in range(4)}

        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=12, restore_best_weights=True, monitor='val_accuracy'),
            keras.callbacks.ReduceLROnPlateau(
                patience=6, factor=0.5, verbose=0),
        ]
        if save_path:
            callbacks.append(keras.callbacks.ModelCheckpoint(
                save_path, save_best_only=True, monitor='val_accuracy',
                verbose=1))

        history = model.fit(
            {'global_view': Xg, 'local_view': Xl},
            y_labels,
            validation_split = val_split,
            epochs           = epochs,
            batch_size       = batch_size,
            callbacks        = callbacks,
            class_weight     = cw,
            verbose          = 1,
        )
        self.model   = model
        self._loaded = True
        return history


# ─────────────────────────────────────────────────────────────────────────────
def _minmax(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.ones_like(arr) * 0.5
    return (arr - mn) / (mx - mn)
