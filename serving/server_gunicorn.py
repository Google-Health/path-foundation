# Copyright 2024 Google LLC
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

"""Launcher for the pete_prediction_executor based encoder server.

Uses the prediction container framework to create a request server which
performs the logic for requests in separate processes and uses a local TFserving
instance to handle the model.
"""

from collections.abc import Sequence
import os

from absl import app
from absl import logging

from prediction_container import server_gunicorn

_EXECUTOR = [
    '/server-env/bin/python3',
    'pete_prediction_executor.py',
    '--logtostderr',
    '--verbosity=1',
]


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if 'AIP_HTTP_PORT' not in os.environ:
    raise ValueError(
        'The environment variable AIP_HTTP_PORT needs to be specified.'
    )
  http_port = int(os.environ.get('AIP_HTTP_PORT'))
  options = {
      'bind': f'0.0.0.0:{http_port}',
      'workers': 3,
      'timeout': 120,
  }
  health_checker = server_gunicorn.ModelServerHealthCheck(
      health_check_port=int(os.environ.get('MODEL_REST_PORT')),
      model_name='default',
  )
  logging.info('Launching gunicorn application.')
  server_gunicorn.PredictionApplication(
      server_gunicorn.SubprocessPredictionExecutor(_EXECUTOR),
      health_check=health_checker,
      options=options,
  ).run()


if __name__ == '__main__':
  app.run(main)
