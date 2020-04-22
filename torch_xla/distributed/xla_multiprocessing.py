from __future__ import division
from __future__ import print_function

import collections
import contextlib
import os
import re
import socket
import sys
import torch.multiprocessing
import torch_xla
import torch_xla.core.xla_env_vars as xenv
import torch_xla.core.xla_model as xm
import traceback

PreForkConfig = collections.namedtuple('PreForkConfig', 'dev_kind num_devices')
WorkerConfigEntry = collections.namedtuple('WorkerConfigEntry',
                                           'worker_name ordinal host_port')
TfDevice = collections.namedtuple('TfDevice', 'job replica task device fqdev')
Device = collections.namedtuple('Device', 'kind index tfdev')

_LOCAL_WORKER = 'localservice'


def _get_free_tcp_ports(n=1):
  ports = []
  for _ in range(0, n):
    with contextlib.closing(socket.socket(socket.AF_INET,
                                          socket.SOCK_STREAM)) as s:
      s.bind(('', 0))
      ports.append(s.getsockname()[1])
  return ports


def _is_xla_config():
  for env in [xenv.TPU_CONFIG, xenv.LOCAL_WORKER, xenv.DEVICE_MAP]:
    if os.environ.get(env, None) is not None:
      return True
  return False


def _create_local_worker_config(index, port, host='localhost'):
  return '{}:{};grpc://{}:{}'.format(_LOCAL_WORKER, index, host, port)


def _create_xla_local_device_name(kind, index, task=0):
  return '/job:{}/replica:0/task:{}/device:XLA_{}:{}'.format(
      _LOCAL_WORKER, task, kind, index)


def _parse_tf_device(devstr):
  m = re.match(r'/job:([^/]+)/replica:(\d+)/task:(\d+)/device:([A-Z_]+:\d+)',
               devstr)
  if not m:
    raise ValueError('Bad device syntax: {}'.format(devstr))
  return TfDevice(
      job=m.group(1),
      replica=int(m.group(2)),
      task=int(m.group(3)),
      device=m.group(4),
      fqdev=m.group(0))


def _parse_device_config(config, select=None):
  # XRT_DEVICE_MAP='GPU:0;/job:localservice/replica:0/task:0/device:XLA_GPU:0|...'
  devices = collections.OrderedDict()
  for devcfg in config.split('|'):
    m = re.match(r'(CPU|TPU|GPU):(\d+);(.+)', devcfg)
    if not m:
      raise ValueError('Bad device syntax: {}'.format(devcfg))
    if select is None or m.group(1) == select:
      tfdev = _parse_tf_device(m.group(3))
      devices['{}:{}'.format(m.group(1), m.group(2))] = Device(
          kind=m.group(1), index=int(m.group(2)), tfdev=tfdev)
  return devices or None


def _generate_gpu_config(num_gpus):
  gpu_devices = []
  for index in range(0, num_gpus):
    devstr = _create_xla_local_device_name('GPU', index)
    tfdev = _parse_tf_device(devstr)
    gpu_devices.append(Device(kind='GPU', index=index, tfdev=tfdev))
  return gpu_devices


def _parse_gpu_env_config():
  num_gpus = os.environ.get(xenv.GPU_NUM_DEVICES, None)
  if num_gpus is not None:
    return _generate_gpu_config(int(num_gpus))
  devmap = os.environ.get(xenv.DEVICE_MAP, None)
  if devmap is not None:
    return _parse_device_config(devmap, select='GPU')


def _parse_tpu_config(config):
  # XRT_TPU_CONFIG='tpu_worker;0;ismz9:25822'
  workers = collections.OrderedDict()
  for worker in config.split('|'):
    m = re.match(r'(\w+);(\d+);([\w.]+:\d+)', worker)
    if not m:
      raise ValueError('Bad worker syntax: {}'.format(worker))
    workers['{}:{}'.format(m.group(1), m.group(2))] = WorkerConfigEntry(
        worker_name=m.group(1), ordinal=int(m.group(2)), host_port=m.group(3))
  return workers


def _get_devices_per_worker():
  if os.environ.get(xenv.TPU_CONFIG, None) is not None:
    return int(os.environ.get(xenv.TPU_NUM_DEVICES, '8')), 'TPU'
  gpu_devices = _parse_gpu_env_config()
  return (len(gpu_devices), 'GPU') if gpu_devices else (1, 'CPU')


def _get_multiprocessing_device():
  return os.environ.get(xenv.MP_DEVICE, None)


def _get_local_worker_index():
  worker = os.environ.get(xenv.LOCAL_WORKER, None)
  if worker is None:
    return 0
  m = re.match(r'(\w+):(\d+)', worker)
  if not m:
    raise ValueError('Bad worker syntax: {}'.format(worker))
  return int(m.group(2))


def _local_index_to_global(index, num_devices):
  return _get_local_worker_index() * num_devices + index


def _setup_world_size(num_devices):
  # We cannot call into xla_model code at this point, as we do not know whether
  # the called code would trigger XLA library initializations (which we must
  # not do at this point). So we avoid calling into xm.xrt_world_size().
  world_size = int(os.environ.get(xenv.WORLD_SIZE, '1')) * num_devices
  os.environ[xenv.WORLD_SIZE] = str(world_size)


def _pre_fork_setup(num_devices):
  dev_count, dev_kind = _get_devices_per_worker()
  if num_devices is None:
    num_devices = dev_count
  elif num_devices not in [1, dev_count]:
    raise ValueError(
        'The number of devices must be either 1 or {}, got {} instead'.format(
            dev_count, num_devices))
  if num_devices > 1 and not os.environ.get(xenv.SERVICE_ADDRESS, None):
    # In multi-processing mode, even if there is only one XLA host, we still
    # bring up the mesh service.
    os.environ[xenv.SERVICE_ADDRESS] = '{}:{}'.format(socket.getfqdn(),
                                                      _get_free_tcp_ports()[0])
  return PreForkConfig(dev_kind=dev_kind, num_devices=num_devices)


def _setup_gpu_worker(index, gindex, pf_cfg):
  os.environ[xenv.MP_DEVICE] = 'GPU:{}'.format(gindex)
  gpu_devices = _parse_gpu_env_config()

  local_worker = 0
  worker_ids = set()
  devices = []
  for i, device in enumerate(gpu_devices.values()):
    if i == gindex:
      local_worker = device.tfdev.task
    worker_ids.add(device.tfdev.task)
    devstr = _create_xla_local_device_name(
        device.kind, device.index, task=device.tfdev.task)
    devices.append('{}:{};{}'.format(device.kind, device.index, devstr))

  workers = []
  for wid in worker_ids:
    port = _get_free_tcp_ports()[0] if wid == local_worker else 0
    workers.append(
        _create_local_worker_config(wid, port, host=socket.getfqdn()))

  os.environ[xenv.WORKERS] = '|'.join(workers)
  os.environ[xenv.DEVICE_MAP] = '|'.join(devices)
  os.environ[xenv.LOCAL_WORKER] = '{}:{}'.format(_LOCAL_WORKER, local_worker)


def _setup_tpu_worker(index, gindex, pf_cfg, tpu_env_config):
  os.environ[xenv.MP_DEVICE] = 'TPU:{}'.format(gindex)
  if xenv.LOCAL_WORKER not in os.environ:
    # The local worker can be missing for a 1 TPU host setup. Make sure we
    # always have one.
    tpu_config = _parse_tpu_config(tpu_env_config)
    worker = list(tpu_config.values())[0]
    os.environ[xenv.LOCAL_WORKER] = '{}:{}'.format(worker.worker_name,
                                                   worker.ordinal)
  if gindex > 0 and xenv.TPU_CONFIG in os.environ:
    # In multi-processing mode, only the process handling the first device of
    # the master worker, will do TPU mesh initialization.
    del os.environ[xenv.TPU_CONFIG]


def _prepare_env_for_index(index, pf_cfg):
  _setup_world_size(pf_cfg.num_devices)
  gindex = _local_index_to_global(index, pf_cfg.num_devices)
  os.environ[xenv.ORDINAL] = str(gindex)
  os.environ[xenv.LOCAL_ORDINAL] = str(index)

  if pf_cfg.dev_kind == 'TPU':
    _setup_tpu_worker(index, gindex, pf_cfg, os.environ[xenv.TPU_CONFIG])
  elif pf_cfg.dev_kind == 'GPU':
    _setup_gpu_worker(index, gindex, pf_cfg)
  return gindex


def _setup_replication():
  if xm.xrt_world_size() > 1:
    device = xm.xla_device()
    xm.set_replication(str(device), [str(device)])


def _start_fn(index, pf_cfg, fn, args):
  gindex = _prepare_env_for_index(index, pf_cfg)
  # Calling _setup_replication() will trigger XLA library initialization, so the
  # environment must be fully setup before doing so.
  _setup_replication()
  exit_code = 0
  try:
    fn(gindex, *args)
  except Exception as e:
    print(
        'Exception in device={}: {}'.format(_get_multiprocessing_device(),
                                            str(e)),
        file=sys.stderr)
    traceback.print_exc(limit=16, file=sys.stderr)
    exit_code = 17
  sys.exit(exit_code)


def spawn(fn,
          args=(),
          nprocs=None,
          join=True,
          daemon=False,
          start_method='spawn'):
  """Enables multi processing based replication.

  Args:
    fn (callable): The function to be called for each device which takes part of
      the replication. The function will be called with a first argument being
      the global index of the process within the replication, followed by the
      arguments passed in `args`.
    args (tuple): The arguments for `fn`.
      Default: Empty tuple
    nprocs (int): The number of processes/devices for the replication. At the
      moment, if specified, can be either 1 or the maximum number of devices.
    join (bool): Whether the call should block waiting for the completion of the
      processes which have being spawned.
      Default: True
    daemon (bool): Whether the processes being spawned should have the `daemon`
      flag set (see Python multi-processing API).
      Default: False
    start_method (string): The Python `multiprocessing` process creation mathod.
      Default: `spawn`

  Returns:
    The same object returned by the `torch.multiprocessing.spawn` API. If
    `nprocs` is 1 the `fn` function will be called directly, and the API will
    not return.
  """
  if not _is_xla_config():
    # If this is not an XLA setup, jump to normal multi-processing.
    nprocs = nprocs or 1
    if nprocs == 1:
      fn(0, *args)
      sys.exit(0)
    else:
      return torch.multiprocessing.spawn(
          fn, args=args, nprocs=nprocs, join=join, daemon=daemon)

  pf_cfg = _pre_fork_setup(nprocs)
  if pf_cfg.num_devices == 1:
    _start_fn(0, pf_cfg, fn, args)
  else:
    return torch.multiprocessing.start_processes(
        _start_fn,
        args=(pf_cfg, fn, args),
        nprocs=pf_cfg.num_devices,
        join=join,
        daemon=daemon,
        start_method=start_method)
