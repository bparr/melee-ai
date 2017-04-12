import argparse
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import os
import pprint
import shutil
import subprocess
import sys
import tempfile
import time


# TODO Get a bunch of remote host identification changed warnings, which can be
#      "solved" by `rm ~/.ssh/known_hosts`. Is there a better solution?
PROJECT = 'melee-ai'  # Can be changed by command line flag!
IMAGE_NAME = 'melee-ai-2017-03-14'
ZONES = ['us-east1-b', 'us-central1-b', 'us-west1-b', 'europe-west1-b',
         'asia-northeast1-b', 'asia-east1-b']
MACHINE_TYPE = 'g1-small'
RUN_SH_FILENAME = 'run.sh'
OUTPUT_DIRNAME = 'outputs'
RSYNC_TIMEOUT_SECONDS = 60 # 1 minute.
WORKER_TIMEOUT_SECONDS = 8.5 * 60  # 8.5 minutes.


# Retrieve metadata on existing instances in a specified zone.
def get_instances(service, zone):
  result = service.instances().list(project=PROJECT, zone=zone).execute()
  if 'items' not in result:
    return dict()

  #pprint.pprint(result['items'])
  return dict((x['name'], x) for x in result['items'])


# Flattens a list of dict into a single dict.
def flat_dicts(l):
  return dict(sum([list(x.items()) for x in l], []))


def get_zone_from_request(request):
  return request['zone'].split('/')[-1]


# Create a new instance, and start it.
def create_instance(service, name, zone):
  # Compute here and not at top of file since command line can change PROJECT.
  source_image = 'projects/%s/global/images/%s' % (PROJECT, IMAGE_NAME)
  instance_body = {
    'name': name,
    'machineType': 'zones/%s/machineTypes/%s' % (zone, MACHINE_TYPE),
    'disks': [{
      'boot': True,
      'autoDelete': True,
      'initializeParams': {'sourceImage': source_image},
    }],
    'networkInterfaces': [{
      'network': 'global/networks/default',
      'accessConfigs': [{'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}],
    }],
  }

  return service.instances().insert(
      project=PROJECT, zone=zone, body=instance_body).execute()


# Used to start a previously stopped (but not deleted) instance.
def start_instance(service, instance_name, zone):
  return service.instances().start(project=PROJECT, zone=zone,
                                   instance=instance_name).execute()


# Stop, but do not delete, an instance.
def stop_instance(service, instance_name, zone):
  return service.instances().stop(project=PROJECT, zone=zone,
                                  instance=instance_name).execute()


# The Google Cloud API is async, so this checks if request is done.
def is_request_done(service, request):
  result = service.zoneOperations().get(project=PROJECT,
                                        zone=get_zone_from_request(request),
                                        operation=request['name']).execute()
  if result['status'] == 'DONE':
    if 'error' in result:
      print('ERROR: ' + str(result['error']))
    return True

  return False


# Get host (string) from instance metadata.
def get_host(instance, gcloud_username):
  nat_ip = instance['networkInterfaces'][0]['accessConfigs'][0]['natIP']
  return gcloud_username + '@' + nat_ip



# Convenience for maintaining an open process with a timeout.
class RunningCommand(object):
  TIMEOUT_RETURN_CODE = -1070342
  TIMEOUT_MESSAGE = 'Command timed out.'

  def __init__(self, popen, timeout_seconds):
    self._popen = popen
    self._end_time = time.time() + timeout_seconds
    # (return code, stdout output, stderr output)
    self._outputs = (None, None, None)

  # Returns true if the command terminated cleanly, or timed out.
  # Returns false if the command is still running.
  def poll(self):
    if self._outputs[0] is not None:
      return True

    if self._popen.poll() is not None:
      stdoutdata, stderrdata = self._popen.communicate()
      self._outputs = (self._popen.returncode, stdoutdata, stderrdata)
      return True

    if time.time() > self._end_time:
      self._popen.terminate()
      self._outputs = (RunningCommand.TIMEOUT_RETURN_CODE,
                       RunningCommand.TIMEOUT_MESSAGE,
                       RunningCommand.TIMEOUT_MESSAGE)
      return True

    return False

  # Undefined behavior if call before poll() returns True.
  # Returns tuple: (return code, stdout output, stderr output)
  def get_outputs(self):
    return self._outputs

  # Undefined behavior if call before poll() returns True.
  # Returns whether the command was successful.
  def was_successful(self):
    return self._outputs[0] == 0

  def stop(self):
    self._popen.terminate()



# Returns a function that takes no arguments and returns host if available,
# otherwise None (e.g. the machine is still starting up).
def create_get_host_fn(service, request, worker_name, gcloud_username):
  def get_host_fn():
    if not is_request_done(service, request):
      return None
    instances = get_instances(service, get_zone_from_request(request))
    if ((not worker_name in instances) or
        (instances[worker_name]['status'] != 'RUNNING')):
      print('ERROR: Started job does not appear ready somehow!')
      return None

    print('Now up and running: ' + worker_name)
    return get_host(instances[worker_name], gcloud_username)

  return get_host_fn



# A single google cloud instance capable of running Melee.
class Worker(object):
  def __init__(self, get_host_fn, local_input_path, local_output_path, git_ref):
    self._get_host_fn = get_host_fn
    self._host = get_host_fn()
    self._local_input_path = local_input_path
    self._local_output_path = local_output_path
    self._git_ref = git_ref

    # Mutable.
    self._job_id = None
    self._running_command = None
    self._temp_path = None
    # List of functions that take no arguments, and return a RunningCommand.
    self._start_command_fns = None


  # Returns whether or not a job just completed.
  def do_work(self):
    if self._host is None:
      self._host = self._get_host_fn()
      if self._host is None:
        # Still no instance to run worker on.
        return False

    if self._job_id is None:
      self._initialize_job()

    if not self._running_command.poll():
      return False

    if not self._running_command.was_successful():
      print(self._running_command.get_outputs())

    if len(self._start_command_fns) > 0:
      self._running_command = self._start_command_fns.pop(0)()
      return False

    shutil.move(os.path.join(self._temp_path, OUTPUT_DIRNAME),
                os.path.join(self._local_output_path, self._job_id))
    # TODO add a check to make sure we didn't skip a lot of frames?
    self._job_id = None
    return True

  # Stop any running processes.
  def stop(self):
    if self._job_id is None:
      return

    self._running_command.stop()
    self._job_id = None


  # Initialize and start new Melee job, queuing all async tasks.
  def _initialize_job(self):
    new_job_id = str(time.time())
    remote_path =  '~/shared/' + new_job_id
    remote_input_path = os.path.join(
        remote_path, os.path.basename(self._local_input_path))
    remote_output_path = os.path.join(remote_path, OUTPUT_DIRNAME)

    # TODO Correctly handle multi-word export values.
    melee_commands = [
      'export MELEE_AI_INPUT_PATH=' + remote_input_path,
      'export MELEE_AI_OUTPUT_PATH=' + remote_output_path,
      'export MELEE_AI_GIT_REF=' + self._git_ref,
      os.path.join(remote_input_path, 'run.sh'),
    ]

    self._job_id = new_job_id
    self._running_command = rsync(
        self._local_input_path, self._host + ':' + remote_path)
    self._temp_path = tempfile.mkdtemp(prefix='melee-ai-' + self._job_id)
    self._start_command_fns = [
        lambda: ssh_to_instance(self._host, melee_commands),
        lambda: rsync(self._host + ':' + remote_output_path, self._temp_path),
    ]



# Returns an rsync RunningCommand.
def rsync(from_path, to_path):
  rsync = subprocess.Popen(
      ['rsync', '-r', '-e',
       'ssh -o StrictHostKeyChecking=no -i ~/.ssh/google_compute_engine',
       from_path, to_path],
      shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  return RunningCommand(rsync, RSYNC_TIMEOUT_SECONDS)



# Returns an ssh RunningCommand to remotely run a list of commands.
def ssh_to_instance(host, command_list):
  # TODO This is a security issue, but only we run commands so ok for now.
  command = ' && '.join(command_list)
  ssh = subprocess.Popen(
      ['ssh', '-oStrictHostKeyChecking=no', '-i',
       '~/.ssh/google_compute_engine', host, command],
      shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  return RunningCommand(ssh, WORKER_TIMEOUT_SECONDS)



def main():
  global PROJECT
  script_directory = os.path.dirname(os.path.realpath(sys.argv[0]))
  parser = argparse.ArgumentParser(description='Run Melee workers.')
  parser.add_argument('-g', '--git-ref', required=True,
                      help='What git branch, hash, etc. to use.')
  parser.add_argument('-i', '--input-directory',
                      default=os.path.join(script_directory, 'inputs/'),
                      help='Directory of input files for melee worker.')
  parser.add_argument('--instances-per-zone', type=int, default=8,
                      help='Max of 8 for free accounts. 23 if "Upgraded."')
  parser.add_argument('--num-games', default=10, type=int,
                      help='Number of melee games to play.')
  parser.add_argument('--num-workers', default=10, type=int,
                      help='Number of worker instances to use.')
  parser.add_argument('-o', '--output-directory',
                      default=os.path.join(script_directory, 'outputs/'),
                      help='Directory to store output files for melee worker.')
  parser.add_argument('-p', '--project', default=PROJECT,
                      help='Google cloud project name.')
  parser.add_argument('-u', '--gcloud-username', required=True,
                      help='gcloud ssh username.')
  parser.add_argument('--worker-instance-prefix',
                      help=('Prefix for all worker instances. Defaults to ' +
                            '--gcloud-username. Used to avoid resusing ' +
                            'instances in two simultaneous trainings.'))

  parser.add_argument('--stop-instances', dest='stop_instances',
                      action='store_true',
                      help='Stop gcloud instances when done. Default.')
  parser.add_argument('--no-stop-instances', dest='stop_instances',
                      action='store_false',
                      help='Do NOT stop gcloud instances when done.')
  parser.set_defaults(stop_instances=True)

  args = parser.parse_args()
  PROJECT = args.project

  # Validate input_directory and output_directory command line flags.
  if not os.path.isdir(args.input_directory):
    raise Exception('--input-directory does not exist')
  if not os.path.isfile(os.path.join(args.input_directory, RUN_SH_FILENAME)):
    raise Exception('--input-directory must contain ' + RUN_SH_FILENAME)
  if not os.path.isdir(args.output_directory):
    raise Exception('--output-directory does not exist')
  local_input_path = os.path.realpath(args.input_directory)
  local_output_path = os.path.realpath(args.output_directory)

  # Validate worker instance prefix.
  instance_prefix = args.worker_instance_prefix or args.gcloud_username
  instance_prefix = instance_prefix.replace('_', '-') + '-melee-ai-'
  if not instance_prefix.replace('-', '').isalnum():
    raise Exception('Worker instance prefix can only conatin lowercase ' +
                    'letters, numbers and hyphens: ' + instance_prefix)

  # Validate have enough resources for num_workers.
  if args.num_workers > args.instances_per_zone * len(ZONES):
    raise Exception('To many workers requested.')

  # Allocate workers to different zones.
  worker_zones = []
  for zone in ZONES:
    if len(worker_zones) == args.num_workers:
      break

    worker_zones += [zone] * (min(args.instances_per_zone,
                                  args.num_workers - len(worker_zones)))

  credentials = GoogleCredentials.get_application_default()
  service = discovery.build('compute', 'v1', credentials=credentials)
  instances = flat_dicts([get_instances(service, z) for z in set(worker_zones)])
  worker_names = [instance_prefix + str(i) + '-' + worker_zones[i]
                  for i in range(args.num_workers)]


  print('Initializing workers (starting instances if needed)...')
  workers = []
  for worker_name, worker_zone in zip(worker_names, worker_zones):
    if not (worker_name in instances):
      create_request = create_instance(service, worker_name, worker_zone)
      get_host_fn = create_get_host_fn(
          service, create_request, worker_name, args.gcloud_username)
      workers.append(Worker(get_host_fn, local_input_path,
                            local_output_path, args.git_ref))
      continue

    instance = instances[worker_name]
    if instance['status'] == 'RUNNING':
      print('Already up and running: ' + worker_name)
      print('Was it EXPECTED to be up and running already???')
      host = get_host(instance, args.gcloud_username)
      workers.append(Worker(lambda: host, local_input_path,
                            local_output_path, args.git_ref))
    elif instance['status'] == 'TERMINATED':
      start_request = start_instance(service, worker_name, worker_zone)
      get_host_fn = create_get_host_fn(
          service, start_request, worker_name, args.gcloud_username)
      workers.append(Worker(get_host_fn, local_input_path,
                            local_output_path, args.git_ref))
    else:
      print('ERROR: Unknown initial instance status: ' + instance['status'])


  print('Running ' + str(args.num_games) + ' games...')
  jobs_completed = 0
  while jobs_completed < args.num_games:
    time.sleep(0.1)
    for worker in workers:
      if worker.do_work():
        jobs_completed += 1
        print('Jobs completed: ' + str(jobs_completed))


  for worker in workers:
    worker.stop()

  if not args.stop_instances:
    return


  print('Stopping workers...')
  stop_requests = []
  for worker_name, worker_zone in zip(worker_names, worker_zones):
    stop_requests.append(stop_instance(service, worker_name, worker_zone))

  requests_remaining = len(stop_requests)
  while requests_remaining > 0:
    time.sleep(1)
    for i, request in enumerate(stop_requests):
      if request is None:
        continue

      if is_request_done(service, request):
        print('Stopped ' + worker_names[i])
        stop_requests[i] = None
        requests_remaining -= 1


if __name__ == '__main__':
  main()
