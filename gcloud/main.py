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

  # TODO assert successful.
  # TODO wait until completion.
  result = service.instances().insert(
      project=PROJECT, zone=ZONE, body=instance_body).execute()


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



class Worker(object):
  # TODO eventually handle a None host, so Worker class first task is to create
  #      the instance.
  def __init__(self, host, local_input_path, local_output_path, git_ref):
    self._host = host
    self._local_input_path = local_input_path
    self._local_output_path = local_output_path
    self._git_ref = git_ref

    # Mutable. Always changed at same time.
    self._running_command = None
    self._job_id = None
    self._remote_output_path = None


  # Returns whether or not a job just completed.
  def do_work(self):
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
  parser.add_argument('-o', '--output-directory',
                      default=os.path.join(script_directory, 'outputs/'),
                      help='Directory to store output files for melee worker.')
  parser.add_argument('-u', '--gcloud-username', required=True,
                      help='gcloud ssh username.')
  args = parser.parse_args()

  # Validate input_directory command line flag.
  if not os.path.isdir(args.input_directory):
    raise Exception('--input-directory does not exist')
  if not os.path.isfile(os.path.join(args.input_directory, RUN_SH_FILENAME)):
    raise Exception('--input-directory must contain ' + RUN_SH_FILENAME)
  if not os.path.isdir(args.output_directory):
    raise Exception('--output-directory does not exist')
  local_input_path = os.path.realpath(args.input_directory)
  local_output_path = os.path.realpath(args.output_directory)


  # TODO autogenerate instance names.
  instance_name = 'melee-ai-2017-03-14-script-test2'

  credentials = GoogleCredentials.get_application_default()
  service = discovery.build('compute', 'v1', credentials=credentials)
  # TODO handle a stopped and non-existent instance.
  # TODO print warning message if already existed!
  #create_instance(service, instance_name)

  instances = get_instances(service)
  instance = instances[instance_name]
  host = instance['networkInterfaces'][0]['accessConfigs'][0]['natIP']
  host = args.gcloud_username + '@' + host


  worker = Worker(host, local_input_path, local_output_path, args.git_ref)
  while not worker.do_work():
    time.sleep(0.1)


if __name__ == '__main__':
  main()
