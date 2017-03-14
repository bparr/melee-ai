import argparse
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import pprint
import subprocess


PROJECT = 'melee-ai'
ZONE = 'us-east1-b'
MACHINE_TYPE = 'zones/%s/machineTypes/g1-small' % ZONE
SOURCE_IMAGE = 'projects/%s/global/images/melee-ai-2017-03-09' % PROJECT


def getInstances(service):
  result = service.instances().list(project=PROJECT, zone=ZONE).execute()
  #pprint.pprint(result['items'])
  return dict((x['name'], x) for x in result['items'])


def createInstance(service, name):
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


def sshToInstance(instance, username):
  host = instance['networkInterfaces'][0]['accessConfigs'][0]['natIP']
  COMMAND = 'ls'  # TODO change.
  # TODO This blocks! Make it not block.
  ssh = subprocess.Popen(
      ['ssh', '-oStrictHostKeyChecking=no', '-i',
       '~/.ssh/google_compute_engine', username + '@' + host, COMMAND],
      shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  result = ssh.stdout.readlines()
  if result == []:
      error = ssh.stderr.readlines()
      print('ERROR: %s' % error)
  else:
      print(result)


def main():
  parser = argparse.ArgumentParser(description='Run Melee workers.')
  parser.add_argument('-u', '--gcloud-username', required=True,
                      help='gcloud ssh username.')
  args = parser.parse_args()

  credentials = GoogleCredentials.get_application_default()
  service = discovery.build('compute', 'v1', credentials=credentials)
  instances = getInstances(service)
  sshToInstance(instances['melee-ai-2017-03-14-script-test2'],
                args.gcloud_username)
  #createInstance(service, 'melee-ai-2017-03-14-script-test2')


if __name__ == '__main__':
  main()
