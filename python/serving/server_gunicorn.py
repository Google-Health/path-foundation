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

Uses the serving framework to create a request server which
performs the logic for requests in separate processes and uses a local TFserving
instance to handle the model.
"""

from collections.abc import Sequence
import contextlib
import os
from typing import Any, Mapping, Dict

from absl import app
from absl import logging
from typing_extensions import override

from serving.serving_framework import inline_prediction_executor
from serving.serving_framework import model_runner
from serving.serving_framework import server_gunicorn
from serving.serving_framework.tensorflow import server_model_runner
from serving import pete_error_mapping
from serving import pete_errors
from serving import pete_logging
from serving import pete_predictor_v2
from serving.data_models import embedding_response
from serving.logging_lib import cloud_logging_client


class PredictionExecutor(inline_prediction_executor.InlinePredictionExecutor):
  """Provides prediction request execution using an inline function."""

  def __init__(self):
    self._pete_predictor = pete_predictor_v2.PetePredictor()
    self._exitstack = contextlib.ExitStack()
    super().__init__(self._run_request, server_model_runner.ServerModelRunner)

  def _run_request(
      self,
      request_json: Mapping[str, Any],
      model_runner: model_runner.ModelRunner,
  ) -> Dict[str, Any]:
    """Runs a single json request using provided components."""
    pete_logging.init_embedding_request_logging()
    try:
      return dict(self._pete_predictor.predict(request_json, model_runner))
    except pete_errors.PeteError as err:
      return dict(
          embedding_response.prediction_error_response_v2(
              pete_error_mapping.get_error_code(err)
          )
      )
    except Exception as err:
      cloud_logging_client.error(
          'Unexpected exception raised while processing request.', err
      )
      raise

  @override
  def start(self):
    pete_logging.init_application_logging()
    self._exitstack.enter_context(self._pete_predictor)
    super().start()


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
      PredictionExecutor(),
      health_check=health_checker,
      options=options,
  ).run()


if __name__ == '__main__':
  app.run(main)
