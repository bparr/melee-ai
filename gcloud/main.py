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
PROJECT = 'melee-ai'
ZONE = 'us-east1-b'
MACHINE_TYPE = 'zones/%s/machineTypes/g1-small' % ZONE
SOURCE_IMAGE = 'projects/%s/global/images/melee-ai-2017-03-14' % PROJECT
RUN_SH_FILENAME = 'run.sh'
OUTPUT_DIRNAME = 'outputs'
WORKER_TIMEOUT_SECONDS = 8.5 * 60  # 8.5 minutes.


def get_instances(service):
  result = service.instances().list(project=PROJECT, zone=ZONE).execute()
  #pprint.pprint(result['items'])
  return dict((x['name'], x) for x in result['items'])


def create_instance(service, name):
  # TODO look into startup_script in instance_body for ssh part.
  instance_body = {
    'name': name,
    'machineType': MACHINE_TYPE,
    'disks': [{
      'boot': True,
      'autoDelete': True,
      'initializeParams': {'sourceImage': SOURCE_IMAGE},
    }],
    'networkInterfaces': [{
      'network': 'global/networks/default',
      'accessConfigs': [{'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}],
    }],
  }

  return service.instances().insert(
      project=PROJECT, zone=ZONE, body=instance_body).execute()

def start_instance(service, instance_name):
  return service.instances().start(project=PROJECT, zone=ZONE,
                                   instance=instance_name).execute()

def stop_instance(service, instance_name):
  return service.instances().stop(project=PROJECT, zone=ZONE,
                                  instance=instance_name).execute()

def is_request_done(service, request):
  result = service.zoneOperations().get(project=PROJECT, zone=ZONE,
                                        operation=request['name']).execute()
  if result['status'] == 'DONE':
    if 'error' in result:
      print('ERROR: ' + str(result['error']))
    return True

  return False

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


def create_get_host_fn(service, request, worker_name, gcloud_username):
  def get_host_fn():
    if not is_request_done(service, request):
      return None
    instances = get_instances(service)
    if ((not worker_name in instances) or
        (instances[worker_name]['status'] != 'RUNNING')):
      print('ERROR: Started job does not appear ready somehow!')
      return None

    print('Now up and running: ' + worker_name)
    return get_host(instances[worker_name], gcloud_username)

  return get_host_fn


class Worker(object):
  def __init__(self, get_host_fn, local_input_path, local_output_path, git_ref):
    self._get_host_fn = get_host_fn
    self._host = get_host_fn()
    self._local_input_path = local_input_path
    self._local_output_path = local_output_path
    self._git_ref = git_ref

    # Mutable. Always changed at same time.
    self._running_command = None
    self._job_id = None
    self._remote_output_path = None


  # Returns whether or not a job just completed.
  def do_work(self):
    if self._host is None:
      self._host = self._get_host_fn()
      if self._host is None:
        # Still no instance to run worker on.
        return False

    # Spawn job on existing machine.
    if self._running_command is None:
      new_job_id = str(time.time())
      remote_path =  '~/shared/' + new_job_id
      rsync(self._local_input_path, self._host + ':' + remote_path)

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

      self._running_command = ssh_to_instance(self._host, melee_commands)
      self._job_id = new_job_id
      self._remote_output_path = remote_output_path


    # Wait for job to complete.
    if not self._running_command.poll():
      return False

    if not self._running_command.was_successful():
      print(self._running_command.get_outputs())

    temp_path = tempfile.mkdtemp(prefix='melee-ai-' + self._job_id)
    rsync(self._host + ':' + self._remote_output_path, temp_path)
    shutil.move(os.path.join(temp_path, OUTPUT_DIRNAME),
                os.path.join(self._local_output_path, self._job_id))
    # TODO add a check to make sure we didn't skip a lot of frames?

    self._running_command = None
    self._job_id = None
    self._remote_output_path = None

    return True



# TODO Make calls to this asynchronous too?
def rsync(from_path, to_path):
  rsync = subprocess.Popen(
      ['rsync', '-r', '-e',
       'ssh -o StrictHostKeyChecking=no -i ~/.ssh/google_compute_engine',
       from_path, to_path],
      shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  stdoutdata, stderrdata = rsync.communicate()
  if rsync.returncode != 0:
    print('rsync had non-zero returncode: %s' % rsync.returncode, file=sys.stderr)
  if stdoutdata:
    print('stdout...')
    print(stdoutdata)
  if stderrdata:
    print('stderr...')
    print(stderrdata)



def ssh_to_instance(host, command_list):
  # TODO This is a security issue, but only we run commands so ok for now.
  command = ' && '.join(command_list)
  ssh = subprocess.Popen(
      ['ssh', '-oStrictHostKeyChecking=no', '-i',
       '~/.ssh/google_compute_engine', host, command],
      shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  return RunningCommand(ssh, WORKER_TIMEOUT_SECONDS)



def main():
  script_directory = os.path.dirname(os.path.realpath(sys.argv[0]))
  parser = argparse.ArgumentParser(description='Run Melee workers.')
  parser.add_argument('-g', '--git-ref', required=True,
                      help='What git branch, hash, etc. to use.')
  parser.add_argument('-i', '--input-directory',
                      default=os.path.join(script_directory, 'inputs/'),
                      help='Directory of input files for melee worker.')
  parser.add_argument('--num-games', default=10, type=int,
                      help='Number of melee games to play.')
  parser.add_argument('--num-workers', default=10, type=int,
                      help='Number of worker instances to use.')
  parser.add_argument('-o', '--output-directory',
                      default=os.path.join(script_directory, 'outputs/'),
                      help='Directory to store output files for melee worker.')
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


  credentials = GoogleCredentials.get_application_default()
  service = discovery.build('compute', 'v1', credentials=credentials)
  instances = get_instances(service)
  worker_names = [instance_prefix + str(i) for i in range(args.num_workers)]


  print('Initializing workers (starting instances if needed)...')
  workers = []
  for worker_name in worker_names:
    if not (worker_name in instances):
      create_request = create_instance(service, worker_name)
      get_host_fn = create_get_host_fn(
          service, create_request, worker_name, args.gcloud_username)
      workers.append(Worker(get_host_fn, local_input_path,
                            local_output_path, args.git_ref))
      continue

    instance = instances[worker_name]
    if instance['status'] == 'RUNNING':
      print('Already up and running: ' + worker_name)
      host = get_host(instance, args.gcloud_username)
      workers.append(Worker(lambda: host, local_input_path,
                            local_output_path, args.git_ref))
    elif instance['status'] == 'TERMINATED':
      start_request = start_instance(service, worker_name)
      get_host_fn = create_get_host_fn(
          service, start_request, worker_name, args.gcloud_username)
      workers.append(Worker(get_host_fn, local_input_path,
                            local_output_path, args.git_ref))
    else:
      print('ERROR: Unknown initial instance status: ' + instance['status'])


  print('Running ' + str(args.num_games) + ' games...')
  jobs_completed = 0
  while jobs_completed < args.num_games:
    for worker in workers:
      if worker.do_work():
        jobs_completed += 1
        print('Jobs complted: ' + str(jobs_completed))


  if not args.stop_instances:
    return

  # TODO Notify the Worker classes (e.g. running command) about stopping?
  print('Stopping workers...')
  stop_requests = [stop_instance(service, x) for x in worker_names]
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
