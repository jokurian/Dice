import os, time
import numpy as np
from numpy.random import Generator, MT19937, PCG64
import scipy as sp
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
#os.environ['JAX_ENABLE_X64'] = 'True'
#os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
import jax.scipy as jsp
from jax import lax, jit, custom_jvp, vmap, random, vjp, checkpoint
from mpi4py import MPI

from functools import partial
print = partial(print, flush=True)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
  print(f'Number of cores: {size}\n', flush=True)


def blocking_analysis(weights, energies, neql, printQ=False):
  nSamples = weights.shape[0] - neql
  weights = weights[neql:]
  energies = energies[neql:]
  weightedEnergies = np.multiply(weights, energies)
  meanEnergy = weightedEnergies.sum() / weights.sum()
  if printQ:
    print(f'\nMean energy: {meanEnergy:.8e}')
    print('Block size    # of blocks        Mean                Error')
  blockSizes = np.array([ 1, 2, 5, 10, 20, 50, 70, 100, 200, 500 ])
  prevError = 0.
  plateauError = None
  for i in blockSizes[blockSizes < nSamples/2.]:
    nBlocks = nSamples//i
    blockedWeights = np.zeros(nBlocks)
    blockedEnergies = np.zeros(nBlocks)
    for j in range(nBlocks):
      blockedWeights[j] = weights[j*i:(j+1)*i].sum()
      blockedEnergies[j] = weightedEnergies[j*i:(j+1)*i].sum() / blockedWeights[j]
    v1 = blockedWeights.sum()
    v2 = (blockedWeights**2).sum()
    mean = np.multiply(blockedWeights, blockedEnergies).sum() / v1
    error = (np.multiply(blockedWeights, (blockedEnergies - mean)**2).sum() / (v1 - v2 / v1) / (nBlocks - 1))**0.5
    if printQ:
      print(f'  {i:4d}           {nBlocks:4d}       {mean:.8e}       {error:.6e}')
    if error < 1.05 * prevError and plateauError is None:
      plateauError = max(error, prevError)
    prevError = error

  if printQ:
    if plateauError is not None:
      print(f'Stocahstic error estimate: {plateauError:.6e}\n')

  return meanEnergy, plateauError


@custom_jvp
def _eigh(a):
    w, v = jnp.linalg.eigh(a)
    return w, v

@_eigh.defjvp
def _eigh_jvp(primals, tangents):
    a = primals[0]
    at = tangents[0]
    w, v = primal_out = _eigh(*primals)

    deg_thresh = 1.e-5
    eji = w[..., np.newaxis, :] - w[..., np.newaxis]
    #idx = abs(eji) < deg_thresh
    #eji = eji.at[idx].set(1.e200)
    eji = jnp.where(eji == 0., 1., eji)
    eji = jnp.where(abs(eji) < deg_thresh, 1.e200, eji)
    #eji = eji.at[jnp.diag_indices_from(eji)].set(1.)
    #eji = eji.at[idx].set(1.e200)
    #eji = eji.at[np.diag_indices_from(eji)].set(1.)
    eye_n = jnp.eye(a.shape[-1])
    Fmat = jnp.reciprocal(eji) - eye_n
    dw, dv = _eigh_jvp_jitted_nob(v, Fmat, at)
    return primal_out, (dw,dv)

@jit
def _eigh_jvp_jitted_nob(v, Fmat, at):
    vt_at_v = jnp.dot(v.conj().T, jnp.dot(at, v))
    dw = jnp.diag(vt_at_v)
    dv = jnp.dot(v, jnp.multiply(Fmat, vt_at_v))
    return dw, dv


#@checkpoint
@jit
def _hf(h1, h2, nelec_proxy):
  h1 = (h1 + h1.T) / 2.
  #print(f'h1:\n{h1}\n')
  #print(f'h2:\n{h2}\n')
  nelec = nelec_proxy.shape[0]
  def scanned_fun(carry, x):
    dm = carry
    #vj = jnp.einsum('ijkl,ji->kl', h2, dm)
    #vk = jnp.einsum('ijkl,jk->il', h2, dm)
    f = jnp.einsum('gij,ik->gjk', h2, dm)
    c = vmap(jnp.trace)(f)
    vj = jnp.einsum('g,gij->ij', c, h2)
    vk = jnp.einsum('glj,gjk->lk', f, h2)
    #print(f'dm:\n{dm}')
    #print(f'f:\n{f}')
    #print(f'c:\n{c}')
    #print(f'vj:\n{vj}')
    #print(f'vk:\n{vk}')
    vhf = vj - 0.5 * vk
    fock = h1 + vhf
    #print(f'fock:\n{fock}')
    #mo_energy, mo_coeff = jnp.linalg.eigh(fock)
    mo_energy, mo_coeff = _eigh(fock)
    #print(f'mo_coeff:\n{mo_coeff}')
    #exit()
    idx = jnp.argmax(abs(mo_coeff.real), axis=0)
    mo_coeff = jnp.where(mo_coeff[idx, jnp.arange(len(mo_energy))].real < 0, -mo_coeff, mo_coeff)
    #mo_coeff[:, mo_coeff[idx, jnp.arange(len(mo_energy))].real < 0] *= -1
    e_idx = jnp.argsort(mo_energy)
    e_sort = mo_energy[e_idx]
    nmo = mo_energy.size
    mo_occ = jnp.zeros(nmo)
    nocc = nelec // 2
    #mo_occ.at[e_idx[:nocc]].set(2)
    #mo_occ = jnp.where(e_idx[:nocc], 2, 0)
    mo_occ = mo_occ.at[e_idx[:nocc]].set(2)
    #mo_occ[e_idx[:nocc]] = 2
    mocc = mo_coeff[:, jnp.nonzero(mo_occ, size=nocc)[0]]
    #mocc = mo_coeff[:, mo_occ > 0]
    dm = (mocc * mo_occ[jnp.nonzero(mo_occ, size=nocc)[0]]).dot(mocc.T)
    #dm = (mocc * mo_occ[mo_occ > 0]).dot(mocc.T)
    return dm, mo_coeff

  norb = h1.shape[0]
  dm0 = 2 * jnp.eye(norb, nelec//2).dot(jnp.eye(norb, nelec//2).T)
  _, mo_coeff = lax.scan(scanned_fun, dm0, None, length=30)

  return mo_coeff[-1]
  #return mo_coeff[-1][2, 0]

#@checkpoint
@jit
def calc_overlap(walker):
  return jnp.linalg.det(walker[:walker.shape[1], :])**2

calc_overlap_vmap = vmap(calc_overlap)


#@checkpoint
@jit
def calc_green(walker):
  return (walker.dot(jnp.linalg.inv(walker[:walker.shape[1], :]))).T

calc_green_vmap = vmap(calc_green)


#@checkpoint
@jit
def calc_force_bias(walker, rot_chol):
  green_walker = calc_green(walker)
  fb = 2. * jnp.einsum('gij,ij->g', rot_chol, green_walker, optimize='optimal')
  return fb

calc_force_bias_vmap = vmap(calc_force_bias, in_axes = (0, None))


#@checkpoint
@jit
def calc_energy(h0, rot_h1, rot_chol, walker):
  ene0 = h0
  green_walker = calc_green(walker)
  ene1 = 2. * jnp.sum(green_walker * rot_h1)
  f = jnp.einsum('gij,jk->gik', rot_chol, green_walker.T, optimize='optimal')
  c = vmap(jnp.trace)(f)
  exc = jnp.sum(vmap(lambda x: x * x.T)(f))
  ene2 = 2. * jnp.sum(c * c) - exc
  return ene2 + ene1 + ene0

calc_energy_vmap = vmap(calc_energy, in_axes = (None, None, None, 0))


# defining this separately because calculating vhs for a batch seems to be faster
#@checkpoint
@jit
def apply_propagator(exp_h1, vhs_i, walker_i):
  walker_i = exp_h1.dot(walker_i)
  def scanned_fun(carry, x):
    carry = vhs_i.dot(carry)
    return carry, carry
  _, vhs_n_walker = lax.scan(scanned_fun, walker_i, jnp.arange(1, 6))
  walker_i = walker_i + jnp.sum(jnp.stack([ vhs_n_walker[n] / np.math.factorial(n+1) for n in range(5) ]), axis=0)
  walker_i = exp_h1.dot(walker_i)
  return walker_i

#@checkpoint
@jit
def apply_propagator_vmap(exp_h1, chol, dt, walkers, fields):
  vhs = 1.j * jnp.sqrt(dt) * fields.dot(chol).reshape(walkers.shape[0], walkers.shape[1], walkers.shape[1])
  return vmap(apply_propagator, in_axes = (None, 0, 0))(exp_h1, vhs, walkers)


# one block of phaseless propagation consisting of nsteps steps followed by energy evaluation
def propagate_phaseless(h0_prop, h0, h1, chol, rot_chol, dt, walkers, weights, e_estimate, mf_shifts, key, nsteps_proxy, nclub_proxy):
#def propagate_phaseless(h0_prop, h0, h1, chol, rot_chol, dt, walkers, weights, e_estimate, mf_shifts, fields, zeta_sr):
  h1 = (h1 + h1.T) / 2.

  nelec_proxy = jnp.zeros((walkers.shape[2]*2))
  mo_coeff = _hf(h1, chol.reshape(-1, h1.shape[0], h1.shape[0]), nelec_proxy)
  #print(mo_coeff)
  #exit()

  h1 = mo_coeff.T.dot(h1).dot(mo_coeff)
  chol = jnp.einsum('gij,jp->gip', chol.reshape(-1, h1.shape[0], h1.shape[0]), mo_coeff)
  chol = jnp.einsum('qi,gip->gqp', mo_coeff.T, chol).reshape(-1, h1.shape[0] * h1.shape[0])
  #nb = C.shape[-1]
  #for i, cv in enumerate(eri):
  #    half = np.dot(cv.reshape(nb, nb), C)
  #    eri[i] = np.dot(C.conj().T, half).ravel()

  #mf_shifts = 2.j * np.array([ np.sum(np.diag(chol_i)[:nelec]) for chol_i in chol])
  #h0_prop = - h0 - np.sum(mf_shifts**2) / 2.
  v0 = 0.5 * jnp.einsum('gik,gjk->ij', chol.reshape(-1, h1.shape[0], h1.shape[0]), chol.reshape(-1, h1.shape[0], h1.shape[0]), optimize='optimal')
  h1_mod = h1 - v0
  h1_mod = h1_mod - jnp.real(1.j * jnp.einsum('g,gik->ik', mf_shifts, chol.reshape(-1, h1.shape[0], h1.shape[0])))
  exp_h1 = jsp.linalg.expm(-dt * h1_mod / 2.)
  rot_h1 = h1[:walkers.shape[2], :].copy()
  rot_chol = chol.reshape(-1, h1.shape[0], h1.shape[0])[:, :walkers.shape[2], :].copy()

  nclub = nclub_proxy.shape[0]
  nsteps = nsteps_proxy.shape[0]

  # carry : [ walkers, weights, overlaps, e_shift ]
  #key = random.PRNGKey(seed)
  #@checkpoint
  def scanned_fun(carry, x):
    #carry[3], subkey = random.split(carry[3])
    #key, subkey = random.split(key)
    #fields = random.normal(subkey, shape=(carry[0].shape[0], chol.shape[0]))
    #fields = random.normal(x, shape=(carry[0].shape[0], chol.shape[0]))
    fields = jnp.array(x)
    force_bias = calc_force_bias_vmap(carry[0], rot_chol)
    field_shifts = -jnp.sqrt(dt) * (1.j * force_bias - mf_shifts)
    shifted_fields = fields - field_shifts
    shift_term = jnp.sum(shifted_fields * mf_shifts, axis=1)
    fb_term = jnp.sum(fields * field_shifts - field_shifts * field_shifts / 2., axis=1)
    #carry[0] = qr_vmap(carry[0])
    #carry[0] = normalize_vmap(carry[0])
    #carry[2] = calc_overlap_vmap(carry[0])
    carry[0] = apply_propagator_vmap(exp_h1, chol, dt, carry[0], shifted_fields)

    overlaps_new = calc_overlap_vmap(carry[0])
    imp_fun = jnp.exp(-jnp.sqrt(dt) * shift_term + fb_term + dt * (carry[3] + h0_prop)) * overlaps_new / carry[2]
    theta = jnp.angle(jnp.exp(-jnp.sqrt(dt) * shift_term) * overlaps_new / carry[2])
    imp_fun_phaseless = jnp.abs(imp_fun) * jnp.cos(theta)
    imp_fun_phaseless = jnp.where(imp_fun_phaseless < 1.e-3, 0., imp_fun_phaseless)
    imp_fun_phaseless = jnp.where(imp_fun_phaseless > 100., 0., imp_fun_phaseless)
    imp_fun_phaseless = jnp.where(jnp.isnan(imp_fun_phaseless), 0., imp_fun_phaseless)
    carry[1] = imp_fun_phaseless * carry[1]
    carry[1] = jnp.where(carry[1] > 100., 0., carry[1])
    carry[2] = overlaps_new
    carry[3] = e_estimate - 0.1 * jnp.log(jnp.sum(carry[1]) / carry[0].shape[0]) / dt
    return carry, x

  #key = random.PRNGKey(seed)
  #keys = random.split(key, nsteps)
  #[ walkers, weights, overlaps, key ], _ = lax.scan(scanned_fun, [ walkers, weights, overlaps, key ], jnp.arange(nsteps))
  #[ walkers, weights, overlaps ], _ = lax.scan(scanned_fun, [ walkers, weights, overlaps ], jnp.arange(nsteps))
  #overlaps = calc_overlap_vmap(walkers)
  #[ walkers, weights, overlaps, _ ], _ = lax.scan(scanned_fun, [ walkers, weights, overlaps, e_estimate ], fields_np)

  # carry : [ walkers, weights, overlaps, e_shift, key ]
  @checkpoint
  def outer_scanned_fun(carry, x):
    carry[4], subkey = random.split(carry[4])
    fields = random.normal(subkey, shape=(nsteps, carry[0].shape[0], chol.shape[0]))
    carry[:4], _ = lax.scan(scanned_fun, carry[:4], fields)
    #carry, _ = lax.scan(scanned_fun, carry, x[0])
    carry[0] = qr_vmap(carry[0])
    energy_samples = jnp.real(calc_energy_vmap(h0, rot_h1, rot_chol, carry[0]))
    energy_samples = jnp.where(jnp.abs(energy_samples - carry[3]) > jnp.sqrt(2./dt), carry[3], energy_samples)
    block_weight = jnp.sum(carry[1])
    block_energy = jnp.sum(energy_samples * carry[1]) / block_weight
    carry[3] = 0.9 * carry[3] + 0.1 * block_energy
    carry[4], subkey = random.split(carry[4])
    zeta = random.uniform(subkey)
    carry[0], carry[1] = stochastic_reconfiguration(carry[0], carry[1], zeta)
    carry[2] = calc_overlap_vmap(carry[0])
    return carry, (block_energy, block_weight)

  #key, subkey = random.split(key)
  #fields = random.normal(subkey, shape=(nclub, nsteps, walkers.shape[0], chol.shape[0]))
  #key, subkey = random.split(key)
  #zeta_sr = random.uniform(subkey, shape=(nclub,))
  overlaps = calc_overlap_vmap(walkers)
  #key = random.PRNGKey(seed)
  [ walkers, weights, overlaps, e_shift, key ], (block_energy, block_weight) = lax.scan(outer_scanned_fun, [ walkers, weights, overlaps, e_estimate, key ], None, length=nclub)
  #[ walkers, weights, overlaps, e_shift ], (block_energy, block_weight) = lax.scan(outer_scanned_fun, [ walkers, weights, overlaps, e_estimate ], (fields, zeta_sr))

  #print(block_energy)
  #exit()
  #return block_energy[-1], (walkers, weights)
  return jnp.sum(block_energy * block_weight) / jnp.sum(block_weight), (walkers, weights, key)

#propagate_phaseless_jit = jit(propagate_phaseless, static_argnums=(11,))
propagate_phaseless_jit = jit(propagate_phaseless)


# called once after each block
#@checkpoint
def qr_vmap(walkers):
  walkers, _ = vmap(jnp.linalg.qr)(walkers)
  return walkers

norm_vmap = vmap(jnp.linalg.norm, in_axes=1)

def normalize(walker):
  return walker / norm_vmap(walker)

normalize_vmap = vmap(normalize)


# this uses numpy but is only called once after each block
def stochastic_reconfiguration_np(walkers, weights, zeta):
  nwalkers = walkers.shape[0]
  walkers = np.array(walkers)
  weights = np.array(weights)
  walkers_new = 0. * walkers
  cumulative_weights = np.cumsum(np.abs(weights))
  total_weight = cumulative_weights[-1]
  average_weight = total_weight / nwalkers
  weights_new = np.ones(nwalkers) * average_weight
  for i in range(nwalkers):
    z = (i + zeta) / nwalkers
    new_i = np.searchsorted(cumulative_weights, z * total_weight)
    walkers_new[i] = walkers[new_i].copy()
  return jnp.array(walkers_new), jnp.array(weights_new)


#@checkpoint
@jit
def stochastic_reconfiguration(walkers, weights, zeta):
  nwalkers = walkers.shape[0]
  cumulative_weights = jnp.cumsum(jnp.abs(weights))
  total_weight = cumulative_weights[-1]
  average_weight = total_weight / nwalkers
  weights = jnp.ones(nwalkers) * average_weight
  z = total_weight * (jnp.arange(nwalkers) + zeta) / nwalkers
  indices = vmap(jnp.searchsorted, in_axes = (None, 0))(cumulative_weights, z)
  walkers = walkers[indices]
  return walkers, weights


# this uses numpy but is only called once after each block
def stochastic_reconfiguration_mpi(walkers, weights, zeta):
  nwalkers = walkers.shape[0]
  walkers = np.array(walkers)
  weights = np.array(weights)
  walkers_new = 0. * walkers
  weights_new = 0. * weights
  global_buffer_walkers = None
  global_buffer_walkers_new = None
  global_buffer_weights = None
  global_buffer_weights_new = None
  if rank == 0:
    global_buffer_walkers = np.zeros((nwalkers * size, walkers.shape[1], walkers.shape[2]), dtype=walkers.dtype)
    global_buffer_walkers_new = np.zeros((nwalkers * size, walkers.shape[1], walkers.shape[2]), dtype=walkers.dtype)
    global_buffer_weights = np.zeros(nwalkers * size, dtype=weights.dtype)
    global_buffer_weights_new = np.zeros(nwalkers * size, dtype=weights.dtype)

  comm.Gather(walkers, global_buffer_walkers, root=0)
  comm.Gather(weights, global_buffer_weights, root=0)

  if rank == 0:
    cumulative_weights = np.cumsum(np.abs(global_buffer_weights))
    total_weight = cumulative_weights[-1]
    average_weight = total_weight / nwalkers / size
    global_buffer_weights_new = (np.ones(nwalkers * size) * average_weight).astype(weights.dtype)
    for i in range(nwalkers * size):
      z = (i + zeta) / nwalkers / size
      new_i = np.searchsorted(cumulative_weights, z * total_weight)
      global_buffer_walkers_new[i] = global_buffer_walkers[new_i].copy()

  comm.Scatter(global_buffer_walkers_new, walkers_new, root=0)
  comm.Scatter(global_buffer_weights_new, weights_new, root=0)
  return jnp.array(walkers_new), jnp.array(weights_new)


def run_afqmc(h0, h1, chol, nelec, dt, nwalkers, nsteps, nblocks, neql=50, seed=0, rdmQ=False, nclub=1):
  init = time.time()
  norb = h1.shape[0]
  nchol = chol.shape[0]

  mf_shifts = 2.j * np.array([ np.sum(np.diag(chol_i)[:nelec]) for chol_i in chol])
  h0_prop = - h0 - np.sum(mf_shifts**2) / 2.
  rot_h1 = h1[:nelec, :].copy()
  rot_chol = chol[:, :nelec, :].copy()
  chol = chol.reshape(nchol, norb * norb)

  mf_shifts = jnp.array(mf_shifts)
  h0_prop = jnp.array(h0_prop)
  h0 = jnp.array(h0)
  rot_h1 = jnp.array(rot_h1)
  h1 = jnp.array(h1)
  chol = jnp.array(chol)
  rot_chol = jnp.array(rot_chol)

  global_block_weights = np.zeros(nblocks)
  local_block_weights = np.zeros(nblocks)
  weights = jnp.ones(nwalkers)
  global_block_weights[0] = nwalkers * size
  local_block_weights[0] = nwalkers
  global_block_energies = np.zeros(nblocks)
  local_block_rdm1 = np.zeros((nblocks, norb, norb))
  walkers = jnp.stack([jnp.eye(norb, nelec) + 0.j for _ in range(nwalkers)])
  energy_samples = jnp.real(calc_energy_vmap(h0, rot_h1, rot_chol, walkers))
  global_block_energies[0] = jnp.sum(energy_samples) / nwalkers   # assuming identical walkers
  e_estimate = jnp.array(global_block_energies[0])
  total_block_energy_n = np.zeros(1, dtype='float32')
  total_block_weight_n = np.zeros(1, dtype='float32')
  #rng = Generator(MT19937(seed + rank))
  key = random.PRNGKey(seed+rank)
  hf_rdm = 2 * np.eye(norb, nelec).dot(np.eye(norb, nelec).T)

  comm.Barrier()
  init_time = time.time() - init
  if rank == 0:
    print("   Iter        Mean energy          Stochastic error       Walltime")
    n = 0
    print(f" {n:5d}      {global_block_energies[0]:.9e}                -              {init_time:.2e} ")
  comm.Barrier()

  local_large_deviations = np.array(0)

  for n in range(1, nblocks):
    #key, subkey = random.split(key)
    #fields = random.normal(subkey, shape=(nclub, nsteps, walkers.shape[0], chol.shape[0]))
    #key, subkey = random.split(key)
    #zeta_sr = random.uniform(subkey, shape=(nclub,))
    #fields = rng.standard_normal(size=(nclub, nsteps, walkers.shape[0], chol.shape[0]))
    #zeta_sr = rng.uniform(size=(nclub,))
    # doing this because of static_argnums not playing nice with vjp
    nsteps_proxy = jnp.zeros((nsteps))
    nclub_proxy = jnp.zeros((nclub))
    if rdmQ:
      block_energy_n, block_vjp, (walkers, weights, key) = vjp(propagate_phaseless_jit, h0_prop, h0, h1, chol, rot_chol, dt, walkers, weights, e_estimate, mf_shifts, key, nsteps_proxy, nclub_proxy, has_aux=True)
      #block_energy_n, block_vjp, (walkers, weights) = vjp(propagate_phaseless_jit, h0_prop, h0, h1, chol, rot_chol, dt, walkers, weights, e_estimate, mf_shifts, fields, zeta_sr, has_aux=True)
      local_block_rdm1[n] = np.array(block_vjp(1.)[2])
      if np.isnan(np.sum(local_block_rdm1[n])) or np.linalg.norm(local_block_rdm1[n] - hf_rdm) > 1.e1 * norb**2  or abs(np.trace(local_block_rdm1[n]) - 2*nelec) > 1.e-3:
        local_block_rdm1[n] = hf_rdm
        local_large_deviations += 1
    else:
      block_energy_n, (walkers, weights, key) = propagate_phaseless_jit(h0_prop, h0, h1, chol, rot_chol, dt, walkers, weights, e_estimate, mf_shifts, key, nsteps_proxy, nclub_proxy)
      #block_energy_n, (walkers, weights) = propagate_phaseless_jit(h0_prop, h0, h1, chol, rot_chol, dt, walkers, weights, e_estimate, mf_shifts, fields, zeta_sr)
    local_block_weights[n] = jnp.sum(weights)
    block_energy_n = np.array([block_energy_n], dtype='float32')
    block_weight_n = np.array([jnp.sum(weights)], dtype='float32')
    block_weighted_energy_n = np.array([block_energy_n * block_weight_n], dtype='float32')
    comm.Reduce([block_weighted_energy_n, MPI.FLOAT], [total_block_energy_n, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce([block_weight_n, MPI.FLOAT], [total_block_weight_n, MPI.FLOAT], op=MPI.SUM, root=0)
    if rank == 0:
      block_weight_n = total_block_weight_n
      block_energy_n = total_block_energy_n / total_block_weight_n
    comm.Bcast(block_weight_n, root=0)
    comm.Bcast(block_energy_n, root=0)
    global_block_weights[n] = block_weight_n
    global_block_energies[n] = block_energy_n
    walkers = qr_vmap(walkers)
    #zeta = rng.uniform()
    key, subkey = random.split(key)
    zeta = random.uniform(subkey)
    walkers, weights = stochastic_reconfiguration_mpi(walkers, weights, zeta)
    e_estimate = 0.9 * e_estimate + 0.1 * global_block_energies[n]

    if n > neql and n%(max(nblocks//10, 1)) == 0:
      comm.Barrier()
      if rank == 0:
        e_afqmc, energy_error = blocking_analysis(global_block_weights[:n], global_block_energies[:n], neql=neql)
        if energy_error is not None:
          print(f" {n:5d}      {e_afqmc:.9e}        {energy_error:.9e}        {time.time() - init:.2e} ", flush=True)
        else:
          print(f" {n:5d}      {e_afqmc:.9e}                -              {time.time() - init:.2e} ", flush=True)
        np.savetxt('samples_jax.dat', np.stack((global_block_weights[:n], global_block_energies[:n])).T)
      comm.Barrier()

  if rdmQ:
    local_weighted_rdm1 = np.array(np.einsum('bij,b->ij', local_block_rdm1[neql:], local_block_weights[neql:]), dtype='float32')
    global_weighted_rdm1 = 0. * local_weighted_rdm1
    global_total_weight = np.zeros(1, dtype='float32')
    global_large_deviations = np.array(0)
    local_total_weight = np.array([np.sum(local_block_weights[neql:])], dtype='float32')
    comm.Reduce([local_weighted_rdm1, MPI.FLOAT], [global_weighted_rdm1, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce([local_total_weight, MPI.FLOAT], [global_total_weight, MPI.FLOAT], op=MPI.SUM, root=0)
    comm.Reduce([local_large_deviations, MPI.INT], [global_large_deviations, MPI.INT], op=MPI.SUM, root=0)
    comm.Barrier()
    with open(f'rdm_{rank}.dat', "w") as fh:
      fh.write(f'{local_total_weight[0]}\n')
      np.savetxt(fh, local_weighted_rdm1 / local_total_weight)
    if rank == 0:
      rdm1 = global_weighted_rdm1 / global_total_weight
      np.savetxt('rdm1_afqmc.txt', rdm1)
      print(f'\nrdm1_tr: {np.trace(rdm1)}', flush=True)
      print(f'\nNumber of large deviations: {global_large_deviations}', flush=True)

  comm.Barrier()
  if rank == 0:
    np.savetxt('samples_jax.dat', np.stack((global_block_weights, global_block_energies)).T)
    e_afqmc, err_afqmc = blocking_analysis(global_block_weights, global_block_energies, neql=neql, printQ=True)
    if err_afqmc is not None:
      sig_dec = int(abs(np.floor(np.log10(err_afqmc))))
      sig_err = np.around(np.round(err_afqmc * 10**sig_dec) * 10**(-sig_dec), sig_dec)
      sig_e = np.around(e_afqmc, sig_dec)
      print(f'AFQMC energy: {sig_e:.{sig_dec}f} +/- {sig_err:.{sig_dec}f}\n')
    elif e_afqmc is not None:
      print(f'AFQMC energy: {e_afqmc}\nCould not find a stochastic error estimate, check blocking analysis\n', flush=True)
  comm.Barrier()

