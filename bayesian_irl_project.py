"""
bayesian_irl_project.py
-----------------------
Online Bayesian IRL from implicit bin-override corrections.

theta = [w_color, w_size, w_shape] on the simplex.
r(obj, bin, theta) is defined in sort_objects_project.py.

After correction (robot→bin_r, human→bin_h):
  P(theta|D) ∝ P(theta) * prod_i Boltzmann(correction_i | theta)

Uses Metropolis-Hastings to maintain a particle approximation.
"""

import numpy as np
from sort_objects_project import all_bin_rewards, correct_bin_under_theta

BETA          = 2.0   # Boltzmann rationality -- higher = sharper likelihood
N_SAMPLES     = 300
N_MH_STEPS    = 40
MH_STD        = 0.04

# Gaussian prior strength: pulls particles toward SIZE_PRIOR after each update.
# Higher = slower learning (more corrections needed to shift theta).
# 0 = no prior (pure likelihood). 20 = strong prior, ~5-6 corrections to flip.
PRIOR_STRENGTH = 20.0
SIZE_PRIOR     = np.array([0.01, 0.98, 0.01])

def _log_prior(theta):
    """Gaussian prior centered on SIZE_PRIOR. Penalizes deviation from size-sorting."""
    return -PRIOR_STRENGTH * float(np.sum((theta - SIZE_PRIOR)**2))


def _softmax(x):
    e = np.exp(BETA * (x - np.max(x)))
    return e / e.sum()


def _log_lik(obj, robot_bin, human_bin, theta):
    """
    Log likelihood of this correction under theta.
    Human chose human_bin when robot chose robot_bin.
    We model: human picks the bin with highest reward under their theta*.
    P(correction | theta) = softmax(rewards)[human_bin]
    """
    rewards  = all_bin_rewards(obj, theta)
    log_prob = np.log(_softmax(rewards) + 1e-12)
    return log_prob[human_bin]


def _project_simplex(v):
    v = np.maximum(v, 0)
    s = v.sum()
    if s < 1e-8:
        return np.ones(3) / 3.0
    return v / s


class BayesianIRL:

    def __init__(self):
        # Robot starts CERTAIN that humans sort by SIZE: theta = [0, 1, 0].
        # Very tight prior -- robot is confidently WRONG at the start.
        # This guarantees ~6/9 conflicts with a color-preferring human in Trial 1.
        # Only human corrections can shift theta toward [1, 0, 0].
        size_biased = np.array([0.01, 0.98, 0.01])
        noise = np.abs(np.random.randn(N_SAMPLES, 3)) * 0.005
        raw   = size_biased + noise
        self.particles = raw / raw.sum(axis=1, keepdims=True)
        self.corrections = []          # list of (obj, robot_bin, human_bin)
        self.total_corrections = 0
        self.corrections_this_trial = 0
        self.prev_entropy = np.log(3)

    def _all_log_likelihoods(self, particles):
        """Compute sum of log likelihoods for all corrections for each particle."""
        scores = np.zeros(len(particles))
        for obj, rb, hb in self.corrections:
            for i, theta in enumerate(particles):
                scores[i] += _log_lik(obj, rb, hb, theta)
        return scores

    def update(self, robot_bin, human_bin, obj):
        """Record a correction and update P(theta|D)."""
        self.corrections.append((obj, robot_bin, human_bin))
        self.total_corrections += 1
        self.corrections_this_trial += 1
        self.prev_entropy = self.get_entropy()

        # reweight: likelihood + prior
        scores = self._all_log_likelihoods(self.particles)
        for i in range(N_SAMPLES):
            scores[i] += _log_prior(self.particles[i])
        lw_max  = scores.max()
        weights = np.exp(scores - lw_max)
        weights /= weights.sum()

        # resample
        idx = np.random.choice(N_SAMPLES, size=N_SAMPLES, p=weights, replace=True)
        self.particles = self.particles[idx].copy()

        # MH diversification
        for _ in range(N_MH_STEPS):
            noise    = np.random.randn(*self.particles.shape) * MH_STD
            proposal = np.array([_project_simplex(self.particles[i] + noise[i])
                                  for i in range(N_SAMPLES)])
            # log acceptance ratio: likelihood ratio + prior ratio
            log_accept = np.zeros(N_SAMPLES)
            for obj2, rb2, hb2 in self.corrections:
                for i in range(N_SAMPLES):
                    log_accept[i] += (_log_lik(obj2, rb2, hb2, proposal[i])
                                    - _log_lik(obj2, rb2, hb2, self.particles[i]))
            # add prior ratio
            for i in range(N_SAMPLES):
                log_accept[i] += _log_prior(proposal[i]) - _log_prior(self.particles[i])
            u      = np.random.rand(N_SAMPLES)
            accept = np.log(u + 1e-12) < log_accept
            self.particles[accept] = proposal[accept]

    def get_map_theta(self):
        # Before any corrections, return the particle with highest prior weight.
        # Since particles are initialized tightly around [0,1,0], this returns
        # the size-sorting policy. Do NOT return uniform [1/3,1/3,1/3].
        if not self.corrections:
            # return the mean of particles -- which is [0.02, 0.96, 0.02]
            return self.particles.mean(axis=0)
        scores  = self._all_log_likelihoods(self.particles)
        best    = np.argmax(scores)
        return self.particles[best].copy()

    def get_mean_theta(self):
        return self.particles.mean(axis=0)

    def get_entropy(self):
        mean = self.get_mean_theta()
        mean = np.clip(mean, 1e-8, 1)
        mean /= mean.sum()
        return float(-np.sum(mean * np.log(mean)))

    def get_best_bin(self, obj):
        """
        Use MEAN theta for robot decisions (not MAP).
        
        Why mean, not MAP?
        - MAP = single highest-scoring particle = can jump to [1,0,0] after 1 correction
        - Mean = average over all particles = changes gradually as posterior shifts
        
        Before any corrections: mean ≈ [0.027, 0.946, 0.027] → sort by SIZE (wrong)
        After correction 1: mean ≈ [0.281, 0.533, 0.186] → still SIZE-dominant
        After correction 2: mean ≈ [0.54, 0.26, 0.20]  → flips to COLOR-dominant
        After corrections 4-5: mean ≈ [0.73, 0.13, 0.14] → converged on COLOR
        
        This gives the gradual learning curve we want for the video demo.
        """
        theta = self.get_mean_theta()
        mode  = "mean theta (size prior)" if not self.corrections else "mean theta (learned)"
        rewards = all_bin_rewards(obj, theta)
        best = int(np.argmax(rewards))
        return best, theta, mode

    def reset_trial(self):
        self.corrections_this_trial = 0
