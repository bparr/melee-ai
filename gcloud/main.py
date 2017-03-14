from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import pprint

PROJECT = 'melee-ai'
ZONE = 'us-east1-b'

def getInstances(compute):
  result = compute.instances().list(project=PROJECT, zone=ZONE).execute()
  return dict((x['name'], x['status']) for x in result['items'])

def main():
  credentials = GoogleCredentials.get_application_default()
  compute = discovery.build('compute', 'v1', credentials=credentials)
  print(getInstances(compute))

if __name__ == '__main__':
  main()
