# Copyright 2022 The VDM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Arguments: [project-id] [zone] [prefix] [machine-from] [machine-to] [machine-type]
# E.g: vdm-xgcp us-central1-a vdm 1 5 v3-8

# Create machines
for i in $(seq $4 $5)
do
  gcloud alpha compute tpus tpu-vm create $3-$i --project=$1 --zone=$2 --accelerator-type=$6 --version=v2-alpha &
done
wait
# Set up machines
for i in $(seq $4 $5)
do
  gcloud alpha compute tpus tpu-vm ssh $3-$i --worker=all --project=$1 --zone=$2 --command "mkdir vdm; gsutil -m rsync -r gs://research-brain-vdm-xgcp/vdm-repo-rsync vdm; bash vdm/sh/setup-vm-tpuv3.sh"&
done
wait

