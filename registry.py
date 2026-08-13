"""Vocabulary registry: multi-mode prototypes over superquadric shape
parameters, with shrinkage-prior variance, spawn/merge, provenance,
grasp-history logging, and multi-part (graph) matching."""
import json
import time
import uuid
import numpy as np

FEATURE_KEYS = ['a1', 'a2', 'eps1', 'eps2', 'a3']
SPAWN_K_SIGMA = 2.5
DEFAULT_INIT_STD = np.array([0.01, 0.01, 0.15, 0.15, 0.01])
MAX_MODES_PER_NOUN = 5
MIN_STD = 1e-4
PRIOR_PSEUDO_N = 4


def canonicalize(params: dict) -> np.ndarray:
    a1, a2 = sorted([params['a1'], params['a2']], reverse=True)
    return np.array([a1, a2, params['eps1'], params['eps2'], params['a3']])


class Mode:
    def __init__(self, mean, m2, n, mode_id=None, grasp_records=None):
        self.mean = mean
        self.m2 = m2
        self.n = n
        self.mode_id = mode_id or uuid.uuid4().hex[:8]
        self.grasp_records = grasp_records or []

    @property
    def std(self) -> np.ndarray:
        prior_var = DEFAULT_INIT_STD ** 2
        if self.n < 2:
            emp_var, emp_n = prior_var.copy(), 0
        else:
            emp_var = self.m2 / (self.n - 1)
            emp_n = self.n - 1
        blended_var = (PRIOR_PSEUDO_N * prior_var + emp_n * emp_var) / (PRIOR_PSEUDO_N + emp_n)
        return np.maximum(np.sqrt(blended_var), MIN_STD)

    def mahalanobis(self, f: np.ndarray) -> float:
        d = (f - self.mean) / self.std
        return float(np.sqrt(np.sum(d ** 2)))

    def membership(self, f: np.ndarray) -> float:
        d = (f - self.mean) / self.std
        return float(np.exp(-0.5 * np.sum(d ** 2)))

    def update(self, f: np.ndarray):
        self.n += 1
        delta = f - self.mean
        self.mean = self.mean + delta / self.n
        delta2 = f - self.mean
        self.m2 = self.m2 + delta * delta2

    def log_grasp(self, approach, contact_region, success):
        self.grasp_records.append({'approach': approach, 'contact_region': contact_region,
                                    'success': bool(success), 'ts': time.time()})

    def grasp_success_rates(self):
        stats = {}
        for rec in self.grasp_records:
            a = rec['approach']
            stats.setdefault(a, {'success': 0, 'total': 0})
            stats[a]['total'] += 1
            if rec['success']:
                stats[a]['success'] += 1
        return {a: v['success'] / v['total'] for a, v in stats.items()}

    def best_grasp_approach(self, min_attempts=3):
        counts = {}
        for rec in self.grasp_records:
            counts.setdefault(rec['approach'], []).append(rec['success'])
        eligible = {a: sum(s) / len(s) for a, s in counts.items() if len(s) >= min_attempts}
        if not eligible:
            return None
        return max(eligible.items(), key=lambda x: x[1])

    def to_dict(self):
        return {'mode_id': self.mode_id, 'mean': self.mean.tolist(), 'm2': self.m2.tolist(),
                'n': self.n, 'grasp_records': self.grasp_records}

    @staticmethod
    def from_dict(d):
        return Mode(mean=np.array(d['mean']), m2=np.array(d['m2']), n=d['n'],
                    mode_id=d['mode_id'], grasp_records=d.get('grasp_records', []))

    @staticmethod
    def bootstrap(f: np.ndarray):
        return Mode(mean=f.copy(), m2=np.zeros_like(f), n=1)


MISSING_PART_PENALTY = 0.4


class GraphMode:
    def __init__(self, mode_id=None):
        self.mode_id = mode_id or uuid.uuid4().hex[:8]
        self.part_modes = {}
        self.relation_dist = {}
        self.n = 0

    def _roles(self, obj_graph):
        return [node.role for node in obj_graph.nodes]

    def structural_distance(self, obj_graph) -> float:
        obs_roles = set(self._roles(obj_graph))
        learned_roles = set(self.part_modes.keys())
        shared = obs_roles & learned_roles
        mismatch = len(obs_roles ^ learned_roles)
        if not shared:
            return 999.0 + mismatch
        node_by_role = {n.role: n for n in obj_graph.nodes}
        dists = [self.part_modes[role].mahalanobis(canonicalize(node_by_role[role].params))
                 for role in shared]
        return float(np.mean(dists)) + mismatch * 2.0

    def membership(self, obj_graph) -> float:
        obs_roles = set(self._roles(obj_graph))
        learned_roles = set(self.part_modes.keys())
        shared = obs_roles & learned_roles
        n_mismatched = len(obs_roles ^ learned_roles)
        if not shared:
            return 0.0
        node_by_role = {n.role: n for n in obj_graph.nodes}
        part_scores = [self.part_modes[role].membership(canonicalize(node_by_role[role].params))
                       for role in shared]
        geo_mean = float(np.prod(part_scores) ** (1.0 / len(part_scores)))
        penalty = MISSING_PART_PENALTY ** n_mismatched
        return geo_mean * penalty

    def update(self, obj_graph):
        self.n += 1
        node_by_role = {n.role: n for n in obj_graph.nodes}
        for role, node in node_by_role.items():
            f = canonicalize(node.params)
            if role not in self.part_modes:
                self.part_modes[role] = Mode.bootstrap(f)
            else:
                self.part_modes[role].update(f)
        for a_id, b_id, rel in obj_graph.edges:
            a_role = next(n.role for n in obj_graph.nodes if n.node_id == a_id)
            b_role = next(n.role for n in obj_graph.nodes if n.node_id == b_id)
            key = tuple(sorted([a_role, b_role]))
            dist = rel['distance']
            if key not in self.relation_dist:
                self.relation_dist[key] = (dist, 0.0, 1)
            else:
                mean, m2, n = self.relation_dist[key]
                n += 1
                delta = dist - mean
                mean += delta / n
                m2 += delta * (dist - mean)
                self.relation_dist[key] = (mean, m2, n)

    def to_dict(self):
        return {'mode_id': self.mode_id, 'n': self.n,
                'part_modes': {role: m.to_dict() for role, m in self.part_modes.items()},
                'relation_dist': {f'{k[0]}|{k[1]}': v for k, v in self.relation_dist.items()}}

    @staticmethod
    def from_dict(d):
        gm = GraphMode(mode_id=d['mode_id'])
        gm.n = d['n']
        gm.part_modes = {role: Mode.from_dict(md) for role, md in d['part_modes'].items()}
        gm.relation_dist = {tuple(k.split('|')): tuple(v) for k, v in d['relation_dist'].items()}
        return gm

    @staticmethod
    def bootstrap(obj_graph):
        gm = GraphMode()
        gm.update(obj_graph)
        return gm


class Registry:
    def __init__(self):
        self.modes = {}
        self.provenance = []
        self.graph_modes = {}

    def classify(self, params, top_k=3):
        """Single-part open-set classification: scores fitted params
        against every learned word's Mode(s), returns top-k (noun,
        confidence). Used for evaluation on datasets with ground-truth
        per-instance masks (e.g. YCB-Video), where blind multi-part
        discovery isn't needed since segmentation is already given."""
        f = canonicalize(params)
        scores = []
        for noun, modes in self.modes.items():
            best = max((m.membership(f) for m in modes), default=0.0)
            scores.append((noun, best))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def match(self, params, noun):
        if noun not in self.modes or not self.modes[noun]:
            return None, None
        f = canonicalize(params)
        scored = [(m.membership(f), m.mode_id) for m in self.modes[noun]]
        return max(scored, key=lambda x: x[0])

    def confirm(self, params, noun, F, crop_ref=None):
        f = canonicalize(params)
        entry = {'ts': time.time(), 'noun': noun, 'F': F, 'features': f.tolist(), 'crop_ref': crop_ref}
        if F != 1:
            entry['action'] = 'logged_only_incorrect'
            self.provenance.append(entry)
            return entry
        modes = self.modes.setdefault(noun, [])
        if not modes:
            new_mode = Mode.bootstrap(f)
            modes.append(new_mode)
            entry['action'] = 'bootstrapped_new_noun'
            entry['mode_id'] = new_mode.mode_id
            self.provenance.append(entry)
            return entry
        dists = [(m.mahalanobis(f), m) for m in modes]
        best_dist, best_mode = min(dists, key=lambda x: x[0])
        if best_dist <= SPAWN_K_SIGMA:
            best_mode.update(f)
            entry['action'] = 'updated_existing_mode'
            entry['mode_id'] = best_mode.mode_id
        else:
            new_mode = Mode.bootstrap(f)
            modes.append(new_mode)
            entry['action'] = 'spawned_new_mode'
            entry['mode_id'] = new_mode.mode_id
        self.provenance.append(entry)
        return entry

    def log_grasp_outcome(self, noun, mode_id, approach, contact_region, success):
        for mode in self.modes.get(noun, []):
            if mode.mode_id == mode_id:
                mode.log_grasp(approach, contact_region, success)
                return True
        return False

    def log_grasp_outcome_graph(self, noun, graph_mode_id, role, approach, contact_region, success):
        for gm in self.graph_modes.get(noun, []):
            if gm.mode_id == graph_mode_id and role in gm.part_modes:
                gm.part_modes[role].log_grasp(approach, contact_region, success)
                return True
        return False

    def match_graph(self, obj_graph, noun):
        gms = self.graph_modes.get(noun)
        if not gms:
            return None, None
        scored = [(gm.membership(obj_graph), gm.mode_id) for gm in gms]
        return max(scored, key=lambda x: x[0])

    def confirm_graph(self, obj_graph, noun, F):
        entry = {'ts': time.time(), 'noun': noun, 'F': F, 'graph': True}
        if F != 1:
            entry['action'] = 'logged_only_incorrect'
            self.provenance.append(entry)
            return entry
        gms = self.graph_modes.setdefault(noun, [])
        if not gms:
            new_gm = GraphMode.bootstrap(obj_graph)
            gms.append(new_gm)
            entry['action'] = 'bootstrapped_new_graph_mode'
            entry['mode_id'] = new_gm.mode_id
            self.provenance.append(entry)
            return entry
        dists = [(gm.structural_distance(obj_graph), gm) for gm in gms]
        best_dist, best_gm = min(dists, key=lambda x: x[0])
        if best_dist <= SPAWN_K_SIGMA:
            best_gm.update(obj_graph)
            entry['action'] = 'updated_existing_graph_mode'
            entry['mode_id'] = best_gm.mode_id
        else:
            new_gm = GraphMode.bootstrap(obj_graph)
            gms.append(new_gm)
            entry['action'] = 'spawned_new_graph_mode'
            entry['mode_id'] = new_gm.mode_id
        self.provenance.append(entry)
        return entry

    def classify_graph(self, obj_graph, top_k=3):
        scores = []
        for noun, gms in self.graph_modes.items():
            best = max((gm.membership(obj_graph) for gm in gms), default=0.0)
            scores.append((noun, best))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def describe(self, noun):
        if noun not in self.modes or not self.modes[noun]:
            return f'I have no learned prototype for "{noun}" yet.'
        lines = [f'I know {len(self.modes[noun])} variant(s) of "{noun}":']
        for m in self.modes[noun]:
            a1, a2, e1, e2, a3 = m.mean
            lines.append(f'  - variant {m.mode_id} (n={m.n}): ~{a1*2000:.0f}x{a2*2000:.0f}mm '
                        f'footprint, {a3*2000:.0f}mm tall, shape exponents ({e1:.2f}, {e2:.2f})')
        return '\n'.join(lines)

    def save(self, path):
        data = {'modes': {noun: [m.to_dict() for m in modes] for noun, modes in self.modes.items()},
                'provenance': self.provenance,
                'graph_modes': {noun: [gm.to_dict() for gm in gms] for noun, gms in self.graph_modes.items()}}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            data = json.load(f)
        reg = Registry()
        reg.modes = {noun: [Mode.from_dict(d) for d in modes] for noun, modes in data['modes'].items()}
        reg.provenance = data['provenance']
        reg.graph_modes = {noun: [GraphMode.from_dict(d) for d in gms]
                            for noun, gms in data.get('graph_modes', {}).items()}
        return reg
