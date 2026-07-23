from brian2 import *
import numpy as np
from config.constants import *
from utils.encoding import gaussian_activation

class CerebellarSNN:
    """
    Cerebellar-inspired spiking neural network.

    Architecture:
        - 9 Mossy Fibers (Poisson, rate-coded inputs)
        - 28 Granule Cells (LIF, fixed sparse connectivity)
        - 2 Purkinje Cells (LIF, agonist-antagonist pair)
        - 2 Climbing Fibers (Poisson, driven by PID error)

    Learning:
        - LTD: weakens synapses when CF fires
        - LTP: homeostatic potentiation when CF is silent
        - Crossed wiring: positive CF -> negative PC, negative CF -> positive PC
    """
    def __init__(self, n_mossy=N_MOSSY, n_granule=N_GRANULE, fanin=GC_FANIN,
                 eta_ltd=ETA_LTD, ltp_rate=LTP_RATE, rng_seed=42):
        self.n_mossy = n_mossy
        self.n_granule = n_granule
        rng = np.random.default_rng(rng_seed)

        # Mossy Fibers 
        self.mossy = PoissonGroup(n_mossy, rates=0 * Hz, name='mossy')

        # Granule Cells 
        gc_eqs = '''dv/dt = (V_RESET - v)/TAU_M : 1 (unless refractory)'''
        self.granule = NeuronGroup(n_granule, gc_eqs, threshold='v>V_THRESH',
                                    reset='v=V_RESET', refractory=T_REF,
                                    method='exact', name='granule')
        self.granule.v = 0

        # Fixed random sparse connectivity: MF -> GC
        pre_idx, post_idx = [], []
        for j in range(n_granule):
            sources = rng.choice(n_mossy, size=fanin, replace=False)
            pre_idx.extend(sources)
            post_idx.extend([j] * fanin)
        self.syn_mossy_gc = Synapses(self.mossy, self.granule, model='w:1',
                                      on_pre='v_post += w', name='mossy_to_gc')
        self.syn_mossy_gc.connect(i=pre_idx, j=post_idx)
        self.syn_mossy_gc.w = 0.45

        # Purkinje Cells 
        pc_eqs = '''
        dv/dt = (V_RESET - v)/TAU_M : 1 (unless refractory)
        dcf/dt = -cf/TAU_CF : 1
        '''
        self.pc = NeuronGroup(2, pc_eqs, threshold='v>V_THRESH',
                               reset='v=V_RESET', refractory=T_REF,
                               method='euler', name='purkinje')
        self.pc.v = 0
        self.pc.cf = 0
        self.pc_pos = self.pc[0:1]
        self.pc_neg = self.pc[1:2]

        # Plasticity: LTD +  LTP 
        pf_pc_model = '''
        dtrace/dt = -trace/(30*ms) : 1 (clock-driven)
        dw/dt = -eta_ltd*w*trace*cf_post/ms + ltp_rate*trace*(1 - cf_post)*(1 - w)/ms : 1 (clock-driven)
        '''
        pf_pc_on_pre = '''
        v_post += w
        trace += 1
        '''
        self.syn_gc_pc = Synapses(self.granule, self.pc, model=pf_pc_model,
                                   on_pre=pf_pc_on_pre, method='euler',
                                   namespace={'eta_ltd': eta_ltd, 'ltp_rate': ltp_rate},
                                   name='gc_to_pc')
        self.syn_gc_pc.connect()
        self.syn_gc_pc.w = 'rand()*0.3 + 0.5'

        # Climbing Fibers 
        self.cf_pos = PoissonGroup(1, rates=0 * Hz, name='cf_pos')
        self.cf_neg = PoissonGroup(1, rates=0 * Hz, name='cf_neg')
        cf_jump = 1.0
        self.syn_cfpos_pcneg = Synapses(self.cf_pos, self.pc_neg,
                                         on_pre=f'cf_post += {cf_jump}',
                                         name='cfpos_to_pcneg')
        self.syn_cfpos_pcneg.connect()
        self.syn_cfneg_pcpos = Synapses(self.cf_neg, self.pc_pos,
                                         on_pre=f'cf_post += {cf_jump}',
                                         name='cfneg_to_pcpos')
        self.syn_cfneg_pcpos.connect()

        
        self.mon_pc = SpikeMonitor(self.pc, name='mon_pc')

        self.net = Network(self.mossy, self.granule, self.pc,
                            self.cf_pos, self.cf_neg,
                            self.syn_mossy_gc, self.syn_gc_pc,
                            self.syn_cfpos_pcneg, self.syn_cfneg_pcpos,
                            self.mon_pc)
        self._prev_pc_counts = np.zeros(2)

    def encode(self, theta, omega, theta_d):
        """Encode state into mossy fiber firing rates."""
        acts = np.concatenate([
            gaussian_activation(theta),
            gaussian_activation(omega),
            gaussian_activation(theta_d),
        ])
        self.mossy.rates = acts * MOSSY_MAX_RATE

    def set_climbing_fibre_rates(self, pid_torque):
        """Set climbing fiber rates based on PID error."""
        pos_drive = np.clip(pid_torque * CF_SCALE, 0, TORQUE_LIMIT) / TORQUE_LIMIT
        neg_drive = np.clip(-pid_torque * CF_SCALE, 0, TORQUE_LIMIT) / TORQUE_LIMIT
        self.cf_pos.rates = pos_drive * CF_MAX_RATE
        self.cf_neg.rates = neg_drive * CF_MAX_RATE

    def step(self, window=CONTROL_DT * second, gain=DECODE_GAIN):
        """Run SNN for one control cycle and return torque."""
        self.net.run(window)
        # Clamp weights to [0, 1] after integration
        self.syn_gc_pc.w = np.clip(self.syn_gc_pc.w[:], 0.0, 1.0)

        counts = self.mon_pc.count[:].astype(float)
        n = counts - self._prev_pc_counts
        self._prev_pc_counts = counts.copy()
        rate_pos, rate_neg = n / window
        torque = gain * (float(rate_pos) - float(rate_neg))
        return np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)

    def reset_dynamic_state(self):
        """Reset internal states (membrane potentials, traces, monitors)."""
        self.pc.v = 0
        self.pc.cf = 0
        self.granule.v = 0
        self.syn_gc_pc.trace = 0
        self.cf_pos.rates = 0 * Hz
        self.cf_neg.rates = 0 * Hz
        self._prev_pc_counts = np.zeros(2)