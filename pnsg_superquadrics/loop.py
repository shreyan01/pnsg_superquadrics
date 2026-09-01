"""Two-source introspective gate (chi vs alpha_t), reusing the RO-MAN
adaptive self-confidence update, plus a graph-aware variant."""
import numpy as np
from registry import Registry

ALPHA_MIN, ALPHA_MAX = 0.05, 0.95
ETA_BASE = 0.05
M_ASSISTED = 0.3
MIN_ABSOLUTE_MU_OBJ = 0.15


class IntrospectiveVocabLoop:
    def __init__(self, registry: Registry = None, alpha0: float = 0.5):
        self.registry = registry or Registry()
        self.alpha = alpha0
        self.history = []

    def _update_alpha(self, F, chi):
        eta = ETA_BASE * M_ASSISTED * (0.5 + self.alpha)
        w = (2 * F - 1) * (1 + chi)
        A = (1 + (1 - self.alpha)) if F == 1 else (1 + self.alpha)
        raw = eta * w * A
        if raw > 0:
            B = (ALPHA_MAX - self.alpha) / (ALPHA_MAX - ALPHA_MIN)
        else:
            B = (self.alpha - ALPHA_MIN) / (ALPHA_MAX - ALPHA_MIN)
        self.alpha = float(np.clip(self.alpha + raw * B, ALPHA_MIN, ALPHA_MAX))

    def step(self, fitted_params, noun, mu_det, feedback_fn):
        mu_obj, mode_id = self.registry.match(fitted_params, noun)
        unknown = mu_obj is None
        if unknown:
            mu_obj_eff, chi = 0.0, 1.0
        else:
            mu_obj_eff = mu_obj
            chi = (mu_det - mu_obj_eff) ** 2
        triggered = unknown or (chi > self.alpha) or (mu_obj_eff < MIN_ABSOLUTE_MU_OBJ)
        record = {'noun': noun, 'mu_det': mu_det, 'mu_obj': mu_obj_eff, 'unknown_noun': unknown,
                  'chi': chi, 'alpha_before': self.alpha, 'triggered': triggered, 'matched_mode': mode_id}
        if triggered:
            F = feedback_fn(noun, fitted_params)
            confirm_entry = self.registry.confirm(fitted_params, noun, F)
            self._update_alpha(F, chi)
            record.update({'F': F, 'registry_action': confirm_entry['action'], 'alpha_after': self.alpha})
        else:
            record.update({'F': None, 'registry_action': 'none_autonomous', 'alpha_after': self.alpha})
        self.history.append(record)
        return record

    def step_graph(self, obj_graph, noun, mu_det, feedback_fn):
        mu_obj, mode_id = self.registry.match_graph(obj_graph, noun)
        unknown = mu_obj is None
        if unknown:
            mu_obj_eff, chi = 0.0, 1.0
        else:
            mu_obj_eff = mu_obj
            chi = (mu_det - mu_obj_eff) ** 2
        triggered = unknown or (chi > self.alpha) or (mu_obj_eff < MIN_ABSOLUTE_MU_OBJ)
        record = {'noun': noun, 'mu_det': mu_det, 'mu_obj': mu_obj_eff, 'unknown_noun': unknown,
                  'chi': chi, 'alpha_before': self.alpha, 'triggered': triggered, 'matched_mode': mode_id}
        if triggered:
            F = feedback_fn(noun, obj_graph)
            confirm_entry = self.registry.confirm_graph(obj_graph, noun, F)
            self._update_alpha(F, chi)
            record.update({'F': F, 'registry_action': confirm_entry['action'], 'alpha_after': self.alpha})
        else:
            record.update({'F': None, 'registry_action': 'none_autonomous', 'alpha_after': self.alpha})
        self.history.append(record)
        return record

    def summary(self):
        n = len(self.history)
        n_trig = sum(1 for r in self.history if r['triggered'])
        return {'n_trials': n, 'n_triggered': n_trig,
                'trigger_rate': n_trig / n if n else 0.0, 'final_alpha': self.alpha}
