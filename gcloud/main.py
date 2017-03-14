import argparse
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import pprint
import subprocess
import sys
import time


PROJECT = 'melee-ai'
ZONE = 'us-east1-b'
MACHINE_TYPE = 'zones/%s/machineTypes/g1-small' % ZONE
SOURCE_IMAGE = 'projects/%s/global/images/melee-ai-2017-03-09' % PROJECT


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
    self._outputs = None  # (return code, stdout output, stderr output)

  # Returns true if the command terminated cleanly, or timed out.
  # Returns false if the command is still running.
  def poll(self):
    if self._outputs is not None:
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



def ssh_to_instance(instance, username):
  host = instance['networkInterfaces'][0]['accessConfigs'][0]['natIP']
  COMMAND = 'sleep 1 && echo hi'  # TODO change.
  # TODO This blocks! Make it not block.
  ssh = subprocess.Popen(
      ['ssh', '-oStrictHostKeyChecking=no', '-i',
       '~/.ssh/google_compute_engine', username + '@' + host, COMMAND],
      shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  #return ssh

  running_command = RunningCommand(ssh, 5.5)
  while not running_command.poll():
    time.sleep(0.1)
  print(running_command.get_outputs())


def main():
  parser = argparse.ArgumentParser(description='Run Melee workers.')
  # TODO add input directory and output directory arguments.
  parser.add_argument('-u', '--gcloud-username', required=True,
                      help='gcloud ssh username.')
  args = parser.parse_args()

  credentials = GoogleCredentials.get_application_default()
  service = discovery.build('compute', 'v1', credentials=credentials)
  instances = get_instances(service)
  instance_name = 'melee-ai-2017-03-14-script-test2'
  ssh_to_instance(instances[instance_name], args.gcloud_username)
  #create_instance(service, 'melee-ai-2017-03-14-script-test2')


if __name__ == '__main__':
  main()
