import json, time, uuid
import numpy as np

FEATURE_KEYS = ['a1', 'a2', 'eps1', 'eps2', 'a3', 'hue', 'saturation',
                'r_10', 'r_30', 'r_50', 'r_70', 'r_90', 'aspect_ratio']
SPAWN_K_SIGMA = 2.5
DEFAULT_INIT_STD = np.array([0.01, 0.01, 0.15, 0.15, 0.01, 40.0, 0.20,
                              0.012, 0.012, 0.012, 0.012, 0.012, 0.4])
MAX_MODES_PER_NOUN = 5
MIN_STD = 1e-4
PRIOR_PSEUDO_N = 4

def canonicalize(params, color_features=None, taper_features=None):
    """color_features: (hue_degrees, saturation) or None.
    taper_features: a 5-tuple of radii (r_10,r_30,r_50,r_70,r_90),
    sampled at 10/30/50/70/90% of the object's height -- a real
    'generalized cylinder' profile, not just a 2-point bottom/top
    comparison. Extended from 2 points to 5 after confirming (real
    synthetic test) that a sharp-step bottle and a smoothly-curved
    (e.g. cola-bottle-shaped) shoulder produce genuinely different
    5-point profiles, a distinction a 2-point measurement blurs away.
    ALWAYS computed for axisymmetric categories, never a discrete
    branch; None only where the concept doesn't apply.
    aspect_ratio = height / max(radius), always computed directly."""
    a1, a2 = sorted([params['a1'], params['a2']], reverse=True)
    hue, sat = color_features if color_features is not None else (0.0, 0.0)
    profile = taper_features if taper_features is not None else (0.0, 0.0, 0.0, 0.0, 0.0)
    aspect_ratio = params['a3'] / max(a1, 1e-6)
    return np.array([a1, a2, params['eps1'], params['eps2'], params['a3'], hue, sat,
                     profile[0], profile[1], profile[2], profile[3], profile[4], aspect_ratio])

def color_active_mask(color_features):
    mask = np.ones(13)
    if color_features is None:
        mask[5] = 0.0; mask[6] = 0.0   # hue, saturation excluded
    return mask

def taper_active_mask(taper_features):
    mask = np.ones(13)
    if taper_features is None:
        mask[7:12] = 0.0   # all 5 profile points excluded
    return mask

def combined_active_mask(color_features, taper_features):
    """The mask callers should actually use -- composes color and taper
    presence independently, since either can be missing without the
    other being affected. aspect_ratio (index 12) is always active."""
    return color_active_mask(color_features) * taper_active_mask(taper_features)

HUE_INDEX = 5   # index of 'hue' in FEATURE_KEYS -- must stay in sync
SHAPE_ONLY_MASK = np.array([1.,1.,1.,1.,1., 0.,0., 0.,0.,0.,0.,0., 1.])
COLOR_ONLY_MASK = np.array([0.,0.,0.,0.,0., 1.,1., 0.,0.,0.,0.,0., 0.])
TAPER_ONLY_MASK = np.array([0.,0.,0.,0.,0., 0.,0., 1.,1.,1.,1.,1., 0.])
GEOMETRY_MASK = np.array([1.,1.,1.,1.,1., 0.,0., 1.,1.,1.,1.,1., 1.])   # shape + profile + aspect, excludes color

def _circular_diff(f, mean):
    """f - mean, but wraps the hue dimension into [-180, 180] first.
    Found necessary: a real object at hue=350deg vs a learned mean of
    hue=10deg is genuinely almost the same color (~20deg apart), but
    raw subtraction gives 340deg -- a massive spurious mismatch that
    tanked accuracy for any color near the wraparound boundary. This
    bug existed in mahalanobis/membership AND in the online mean update
    (Welford's delta), silently corrupting the learned hue mean over
    time, not just single comparisons."""
    diff = f - mean
    diff = diff.copy()
    diff[HUE_INDEX] = ((diff[HUE_INDEX] + 180) % 360) - 180
    return diff

class Mode:
    def __init__(self, mean, m2, n, mode_id=None, grasp_records=None):
        self.mean = mean; self.m2 = m2; self.n = n
        self.mode_id = mode_id or uuid.uuid4().hex[:8]
        self.grasp_records = grasp_records or []
    @property
    def std(self):
        prior_var = DEFAULT_INIT_STD ** 2
        if self.n < 2:
            emp_var, emp_n = prior_var.copy(), 0
        else:
            emp_var = self.m2 / (self.n - 1); emp_n = self.n - 1
        blended_var = (PRIOR_PSEUDO_N * prior_var + emp_n * emp_var) / (PRIOR_PSEUDO_N + emp_n)
        return np.maximum(np.sqrt(blended_var), MIN_STD)
    def mahalanobis(self, f, active_mask=None):
        d = _circular_diff(f, self.mean) / self.std
        if active_mask is not None:
            d = d * active_mask
        return float(np.sqrt(np.sum(d ** 2)))
    def membership(self, f, active_mask=None):
        d = _circular_diff(f, self.mean) / self.std
        if active_mask is not None:
            d = d * active_mask
        return float(np.exp(-0.5 * np.sum(d ** 2)))
    def update(self, f):
        self.n += 1
        delta = _circular_diff(f, self.mean)
        self.mean = self.mean + delta / self.n
        self.mean[HUE_INDEX] = self.mean[HUE_INDEX] % 360.0   # keep the mean itself in [0,360)
        delta2 = _circular_diff(f, self.mean)
        self.m2 = self.m2 + delta * delta2
    def log_grasp(self, approach, contact_region, success):
        self.grasp_records.append({'approach': approach, 'contact_region': contact_region,
                                    'success': bool(success), 'ts': time.time()})
    def grasp_success_rates(self):
        stats = {}
        for rec in self.grasp_records:
            a = rec['approach']; stats.setdefault(a, {'success': 0, 'total': 0})
            stats[a]['total'] += 1
            if rec['success']: stats[a]['success'] += 1
        return {a: v['success'] / v['total'] for a, v in stats.items()}
    def best_grasp_approach(self, min_attempts=3):
        counts = {}
        for rec in self.grasp_records:
            counts.setdefault(rec['approach'], []).append(rec['success'])
        eligible = {a: sum(s) / len(s) for a, s in counts.items() if len(s) >= min_attempts}
        if not eligible: return None
        return max(eligible.items(), key=lambda x: x[1])
    def to_dict(self):
        return {'mode_id': self.mode_id, 'mean': self.mean.tolist(), 'm2': self.m2.tolist(),
                'n': self.n, 'grasp_records': self.grasp_records}
    @staticmethod
    def from_dict(d):
        return Mode(mean=np.array(d['mean']), m2=np.array(d['m2']), n=d['n'],
                    mode_id=d['mode_id'], grasp_records=d.get('grasp_records', []))
    @staticmethod
    def bootstrap(f):
        return Mode(mean=f.copy(), m2=np.zeros_like(f), n=1)

MISSING_PART_PENALTY = 0.4
CONFIDENCE_PSEUDO_N = 8   # a mode's final score gets multiplied by n/(n+8) -- a thin,
                          # poorly-estimated mode (e.g. n=1-3) doesn't just get a wider,
                          # more forgiving variance, it also gets systematically LESS
                          # TRUSTED in the final head-to-head decision. Found necessary
                          # after splitting bottle into two sub-words: the new words'
                          # thin modes (avg n~3-4) were winning argmax comparisons
                          # against can's well-supported modes (n=12-21) purely by
                          # being too permissive (a wide Gaussian assigns nonzero
                          # density broadly, so it doesn't punish deviation the way a
                          # tight, correctly-confident Gaussian does), not by being
                          # genuinely more correct.

def confidence_discount(n):
    return n / (n + CONFIDENCE_PSEUDO_N)



class GraphMode:
    def __init__(self, mode_id=None):
        self.mode_id = mode_id or uuid.uuid4().hex[:8]
        self.part_modes = {}; self.relation_dist = {}; self.n = 0
    def _roles(self, obj_graph):
        return [node.role for node in obj_graph.nodes]
    def structural_distance(self, obj_graph):
        obs_roles = set(self._roles(obj_graph)); learned_roles = set(self.part_modes.keys())
        shared = obs_roles & learned_roles; mismatch = len(obs_roles ^ learned_roles)
        if not shared: return 999.0 + mismatch
        node_by_role = {n.role: n for n in obj_graph.nodes}
        dists = [self.part_modes[role].mahalanobis(
                    canonicalize(node_by_role[role].params, node_by_role[role].color_features,
                                 getattr(node_by_role[role], 'taper_features', None)),
                    combined_active_mask(node_by_role[role].color_features,
                                         getattr(node_by_role[role], 'taper_features', None)))
                 for role in shared]
        return float(np.mean(dists)) + mismatch * 2.0
    def membership(self, obj_graph):
        """Dominant part = primary evidence; secondary parts = bonus
        modifier (can boost, never zero out). Fixes a confirmed real
        bug: strict geometric mean let one undertrained secondary part
        (e.g. n=2, underflowed to 0.0000) erase an otherwise-strong
        dominant match (0.478), losing to a worse-but-luckier competitor.
        Verified against the actual traced case before landing this fix.
        Feature vector now includes hue/saturation AND continuous taper
        (r_bottom, r_top) -- taper is ALWAYS measured, never a discrete
        branch, after a discrete neck/no-neck decision was found to
        regress real accuracy under single-frame sensor noise. Missing
        color or taper data is genuinely EXCLUDED from scoring via the
        active mask, not compared against the learned mean as data."""
        obs_roles = set(self._roles(obj_graph)); learned_roles = set(self.part_modes.keys())
        shared = obs_roles & learned_roles; n_mismatched = len(obs_roles ^ learned_roles)
        if not shared: return 0.0
        node_by_role = {n.role: n for n in obj_graph.nodes}
        if 'dominant' in shared:
            dnode = node_by_role['dominant']
            dtaper = getattr(dnode, 'taper_features', None)
            dominant_score = self.part_modes['dominant'].membership(
                canonicalize(dnode.params, dnode.color_features, dtaper),
                combined_active_mask(dnode.color_features, dtaper))
            modifier = 1.0
            for role in shared:
                if role == 'dominant': continue
                snode = node_by_role[role]
                staper = getattr(snode, 'taper_features', None)
                sec_score = self.part_modes[role].membership(
                    canonicalize(snode.params, snode.color_features, staper),
                    combined_active_mask(snode.color_features, staper))
                modifier *= (0.6 + 0.4 * sec_score)
            geo_mean = dominant_score * modifier
        else:
            part_scores = [self.part_modes[role].membership(
                            canonicalize(node_by_role[role].params, node_by_role[role].color_features,
                                         getattr(node_by_role[role], 'taper_features', None)),
                            combined_active_mask(node_by_role[role].color_features,
                                                 getattr(node_by_role[role], 'taper_features', None)))
                           for role in shared]
            geo_mean = float(np.prod(part_scores) ** (1.0 / len(part_scores)))
        penalty = MISSING_PART_PENALTY ** n_mismatched
        return geo_mean * penalty * confidence_discount(self.n)
    def membership_ensembled(self, obj_graph):
        """EXPERIMENTAL, separate from membership() -- scores geometry
        (shape AND taper, both real geometric measurements) and color
        as fully independent sub-models on the dominant part, then
        combines them with geometry as primary evidence and color as a
        bounded bonus (0.6-1.0x), the same philosophy already proven
        for combining dominant/secondary PARTS, now applied one level
        up to combining GEOMETRY and COLOR. Built after finding that
        the joint 7D scoring gave bottle an AUC of 0.327 -- worse than
        random -- suggesting the combined feature vector was actively
        letting color corrupt an otherwise-usable shape signal."""
        obs_roles = set(self._roles(obj_graph)); learned_roles = set(self.part_modes.keys())
        shared = obs_roles & learned_roles; n_mismatched = len(obs_roles ^ learned_roles)
        if not shared or 'dominant' not in shared: return 0.0
        node_by_role = {n.role: n for n in obj_graph.nodes}
        dnode = node_by_role['dominant']
        dtaper = getattr(dnode, 'taper_features', None)
        f = canonicalize(dnode.params, dnode.color_features, dtaper)
        mode = self.part_modes['dominant']
        geo_mask = GEOMETRY_MASK * taper_active_mask(dtaper)
        geo_score = mode.membership(f, geo_mask)
        if dnode.color_features is not None:
            color_score = mode.membership(f, COLOR_ONLY_MASK)
        else:
            color_score = 0.5   # neutral prior -- neither boosts nor penalizes when color is missing
        dominant_score = geo_score * (0.6 + 0.4 * color_score)
        modifier = 1.0
        for role in shared:
            if role == 'dominant': continue
            snode = node_by_role[role]
            staper = getattr(snode, 'taper_features', None)
            sec_score = self.part_modes[role].membership(
                canonicalize(snode.params, snode.color_features, staper),
                combined_active_mask(snode.color_features, staper))
            modifier *= (0.6 + 0.4 * sec_score)
        penalty = MISSING_PART_PENALTY ** n_mismatched
        return dominant_score * modifier * penalty * confidence_discount(self.n)
    def update(self, obj_graph):
        """Note: unlike color (which can randomly fail per-frame even
        for the same object), taper presence is tied to the WORD's
        category (axisymmetric or not), so it's consistently present
        or consistently absent within any given mode's training
        examples -- no risk of a placeholder-0 example corrupting a
        running mean that's otherwise built from real values, so no
        masking is needed here (masking during MATCHING still matters,
        handled in membership()/structural_distance() above)."""
        self.n += 1
        node_by_role = {n.role: n for n in obj_graph.nodes}
        for role, node in node_by_role.items():
            f = canonicalize(node.params, node.color_features, getattr(node, 'taper_features', None))
            if role not in self.part_modes: self.part_modes[role] = Mode.bootstrap(f)
            else: self.part_modes[role].update(f)
        for a_id, b_id, rel in obj_graph.edges:
            a_role = next(n.role for n in obj_graph.nodes if n.node_id == a_id)
            b_role = next(n.role for n in obj_graph.nodes if n.node_id == b_id)
            key = tuple(sorted([a_role, b_role])); dist = rel['distance']
            if key not in self.relation_dist: self.relation_dist[key] = (dist, 0.0, 1)
            else:
                mean, m2, n = self.relation_dist[key]; n += 1
                delta = dist - mean; mean += delta / n; m2 += delta * (dist - mean)
                self.relation_dist[key] = (mean, m2, n)
    def merge_with(self, other):
        merged = GraphMode(); merged.n = self.n + other.n
        all_roles = set(self.part_modes.keys()) | set(other.part_modes.keys())
        for role in all_roles:
            a = self.part_modes.get(role); b = other.part_modes.get(role)
            if a is not None and b is not None:
                n_total = a.n + b.n
                merged_mean = (a.mean * a.n + b.mean * b.n) / n_total
                var_a = a.m2 / max(a.n - 1, 1); var_b = b.m2 / max(b.n - 1, 1)
                pooled_var = ((a.n - 1) * var_a + (b.n - 1) * var_b) / max(n_total - 2, 1)
                pooled_var += (a.n * b.n / n_total) * ((a.mean - b.mean) ** 2) / n_total
                merged.part_modes[role] = Mode(mean=merged_mean, m2=pooled_var * max(n_total - 1, 1), n=n_total)
            else:
                merged.part_modes[role] = a if a is not None else b
        merged.relation_dist = self.relation_dist if self.n >= other.n else other.relation_dist
        return merged
    def to_dict(self):
        return {'mode_id': self.mode_id, 'n': self.n,
                'part_modes': {role: m.to_dict() for role, m in self.part_modes.items()},
                'relation_dist': {f'{k[0]}|{k[1]}': v for k, v in self.relation_dist.items()}}
    @staticmethod
    def from_dict(d):
        gm = GraphMode(mode_id=d['mode_id']); gm.n = d['n']
        gm.part_modes = {role: Mode.from_dict(md) for role, md in d['part_modes'].items()}
        gm.relation_dist = {tuple(k.split('|')): tuple(v) for k, v in d['relation_dist'].items()}
        return gm
    @staticmethod
    def bootstrap(obj_graph):
        gm = GraphMode(); gm.update(obj_graph); return gm

class Registry:
    def __init__(self):
        self.modes = {}; self.provenance = []; self.graph_modes = {}
    def classify(self, params, top_k=3, color_features=None):
        f = canonicalize(params, color_features); mask = color_active_mask(color_features); scores = []
        for noun, modes in self.modes.items():
            best = max((m.membership(f, mask) for m in modes), default=0.0)
            scores.append((noun, best))
        scores.sort(key=lambda x: -x[1]); return scores[:top_k]
    def match(self, params, noun, color_features=None):
        if noun not in self.modes or not self.modes[noun]: return None, None
        f = canonicalize(params, color_features); mask = color_active_mask(color_features)
        scored = [(m.membership(f, mask), m.mode_id) for m in self.modes[noun]]
        return max(scored, key=lambda x: x[0])
    def confirm(self, params, noun, F, crop_ref=None, color_features=None):
        f = canonicalize(params, color_features)
        entry = {'ts': time.time(), 'noun': noun, 'F': F, 'features': f.tolist(), 'crop_ref': crop_ref}
        if F != 1:
            entry['action'] = 'logged_only_incorrect'; self.provenance.append(entry); return entry
        modes = self.modes.setdefault(noun, [])
        if not modes:
            new_mode = Mode.bootstrap(f); modes.append(new_mode)
            entry['action'] = 'bootstrapped_new_noun'; entry['mode_id'] = new_mode.mode_id
            self.provenance.append(entry); return entry
        dists = [(m.mahalanobis(f), m) for m in modes]
        best_dist, best_mode = min(dists, key=lambda x: x[0])
        if best_dist <= SPAWN_K_SIGMA:
            best_mode.update(f); entry['action'] = 'updated_existing_mode'; entry['mode_id'] = best_mode.mode_id
        else:
            new_mode = Mode.bootstrap(f); modes.append(new_mode)
            entry['action'] = 'spawned_new_mode'; entry['mode_id'] = new_mode.mode_id
            self._maybe_merge(noun)
        self.provenance.append(entry); return entry
    def _maybe_merge(self, noun):
        modes = self.modes[noun]
        if len(modes) <= MAX_MODES_PER_NOUN: return
        best_pair, best_dist = None, np.inf
        for i in range(len(modes)):
            for j in range(i + 1, len(modes)):
                d = np.linalg.norm(modes[i].mean - modes[j].mean)
                if d < best_dist: best_dist, best_pair = d, (i, j)
        i, j = best_pair; a, b = modes[i], modes[j]
        n_total = a.n + b.n
        merged_mean = (a.mean * a.n + b.mean * b.n) / n_total
        var_a = a.m2 / max(a.n - 1, 1); var_b = b.m2 / max(b.n - 1, 1)
        pooled_var = ((a.n - 1) * var_a + (b.n - 1) * var_b) / max(n_total - 2, 1)
        pooled_var += (a.n * b.n / n_total) * ((a.mean - b.mean) ** 2) / n_total
        merged = Mode(mean=merged_mean, m2=pooled_var * max(n_total - 1, 1), n=n_total)
        new_modes = [m for k, m in enumerate(modes) if k not in (i, j)]; new_modes.append(merged)
        self.modes[noun] = new_modes
    def log_grasp_outcome(self, noun, mode_id, approach, contact_region, success):
        for mode in self.modes.get(noun, []):
            if mode.mode_id == mode_id: mode.log_grasp(approach, contact_region, success); return True
        return False
    def log_grasp_outcome_graph(self, noun, graph_mode_id, role, approach, contact_region, success):
        for gm in self.graph_modes.get(noun, []):
            if gm.mode_id == graph_mode_id and role in gm.part_modes:
                gm.part_modes[role].log_grasp(approach, contact_region, success); return True
        return False
    def match_graph(self, obj_graph, noun):
        gms = self.graph_modes.get(noun)
        if not gms: return None, None
        scored = [(gm.membership(obj_graph), gm.mode_id) for gm in gms]
        return max(scored, key=lambda x: x[0])
    def confirm_graph(self, obj_graph, noun, F):
        entry = {'ts': time.time(), 'noun': noun, 'F': F, 'graph': True}
        if F != 1:
            entry['action'] = 'logged_only_incorrect'; self.provenance.append(entry); return entry
        gms = self.graph_modes.setdefault(noun, [])
        if not gms:
            new_gm = GraphMode.bootstrap(obj_graph); gms.append(new_gm)
            entry['action'] = 'bootstrapped_new_graph_mode'; entry['mode_id'] = new_gm.mode_id
            self.provenance.append(entry); return entry
        dists = [(gm.structural_distance(obj_graph), gm) for gm in gms]
        best_dist, best_gm = min(dists, key=lambda x: x[0])
        if best_dist <= SPAWN_K_SIGMA:
            best_gm.update(obj_graph); entry['action'] = 'updated_existing_graph_mode'; entry['mode_id'] = best_gm.mode_id
        else:
            new_gm = GraphMode.bootstrap(obj_graph); gms.append(new_gm)
            entry['action'] = 'spawned_new_graph_mode'; entry['mode_id'] = new_gm.mode_id
            self._maybe_merge_graph(noun)
        self.provenance.append(entry); return entry
    def _maybe_merge_graph(self, noun):
        gms = self.graph_modes[noun]
        if len(gms) <= MAX_MODES_PER_NOUN: return
        best_pair, best_dist = None, np.inf
        for i in range(len(gms)):
            for j in range(i + 1, len(gms)):
                shared = set(gms[i].part_modes.keys()) & set(gms[j].part_modes.keys())
                if not shared: continue
                dists = [gms[i].part_modes[r].mahalanobis(gms[j].part_modes[r].mean) for r in shared]
                d = float(np.mean(dists))
                if d < best_dist: best_dist, best_pair = d, (i, j)
        if best_pair is None: return
        i, j = best_pair; merged = gms[i].merge_with(gms[j])
        new_gms = [gm for k, gm in enumerate(gms) if k not in (i, j)]; new_gms.append(merged)
        self.graph_modes[noun] = new_gms
    def classify_graph(self, obj_graph, top_k=3):
        scores = []
        for noun, gms in self.graph_modes.items():
            best = max((gm.membership(obj_graph) for gm in gms), default=0.0)
            scores.append((noun, best))
        scores.sort(key=lambda x: -x[1]); return scores[:top_k]
    def classify_graph_ensembled(self, obj_graph, top_k=3):
        """EXPERIMENTAL: same as classify_graph but scores via
        membership_ensembled() -- geometry primary, color as a bounded
        bonus -- instead of the joint 7D combined score."""
        scores = []
        for noun, gms in self.graph_modes.items():
            best = max((gm.membership_ensembled(obj_graph) for gm in gms), default=0.0)
            scores.append((noun, best))
        scores.sort(key=lambda x: -x[1]); return scores[:top_k]
    def describe_graph(self, noun):
        gms = self.graph_modes.get(noun)
        if not gms: return f'I have no learned graph structure for "{noun}" yet.'
        lines = [f'I know {len(gms)} structural variant(s) of "{noun}":']
        for gm in gms:
            roles = list(gm.part_modes.keys())
            lines.append(f'  - graph mode {gm.mode_id} (n={gm.n}): parts={roles}')
            for role, m in gm.part_modes.items():
                a1, a2, e1, e2, a3, hue, sat, r10, r30, r50, r70, r90, ratio = m.mean
                profile_str = ','.join(f'{r*1000:.0f}' for r in [r10,r30,r50,r70,r90])
                lines.append(f'      {role}: ~{a1*2000:.0f}x{a2*2000:.0f}mm footprint, {a3*2000:.0f}mm tall, '
                            f'eps=({e1:.2f},{e2:.2f}), color=(hue={hue:.0f}deg,sat={sat:.2f}), '
                            f'profile=[{profile_str}]mm, aspect_ratio={ratio:.2f} [n={m.n}]')
        return '\n'.join(lines)
    def describe(self, noun):
        if noun not in self.modes or not self.modes[noun]: return f'I have no learned prototype for "{noun}" yet.'
        lines = [f'I know {len(self.modes[noun])} variant(s) of "{noun}":']
        for m in self.modes[noun]:
            a1, a2, e1, e2, a3, hue, sat, r10, r30, r50, r70, r90, ratio = m.mean
            profile_str = ','.join(f'{r*1000:.0f}' for r in [r10,r30,r50,r70,r90])
            lines.append(f'  - variant {m.mode_id} (n={m.n}): ~{a1*2000:.0f}x{a2*2000:.0f}mm footprint, {a3*2000:.0f}mm tall, '
                        f'shape exponents ({e1:.2f}, {e2:.2f}), color=(hue={hue:.0f}deg,sat={sat:.2f}), '
                        f'profile=[{profile_str}]mm, aspect_ratio={ratio:.2f}')
        return '\n'.join(lines)
    def save(self, path):
        data = {'modes': {noun: [m.to_dict() for m in modes] for noun, modes in self.modes.items()},
                'provenance': self.provenance,
                'graph_modes': {noun: [gm.to_dict() for gm in gms] for noun, gms in self.graph_modes.items()}}
        with open(path, 'w') as f: json.dump(data, f, indent=2)
    @staticmethod
    def load(path):
        with open(path) as f: data = json.load(f)
        reg = Registry()
        reg.modes = {noun: [Mode.from_dict(d) for d in modes] for noun, modes in data['modes'].items()}
        reg.provenance = data['provenance']
        reg.graph_modes = {noun: [GraphMode.from_dict(d) for d in gms] for noun, gms in data.get('graph_modes', {}).items()}
        return reg