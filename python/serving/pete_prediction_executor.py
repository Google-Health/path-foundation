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

"""Executable to carry out pathology encoding request glue code.

A subprocess which handles a piped in pathology encoder endpoint request json
body and returns the response json body to stdout. Depends on a local TFserving
instance to provide the encoder model.
"""

from collections.abc import Sequence
import json
import sys
import time
from typing import Any, Mapping

from absl import app

from serving.serving_framework.tensorflow import server_model_runner
from serving import abstract_pete_predictor
from serving import pete_error_mapping
from serving import pete_errors
from serving import pete_logging
from serving import pete_predictor_v2
from serving.data_models import embedding_response
from serving.logging_lib import cloud_logging_client


def _run_request(
    request_str: str,
    predictor: abstract_pete_predictor.AbstractPetePredictor,
    model_runner: server_model_runner.ServerModelRunner,
) -> Mapping[str, Any]:
  """Runs a single json request using provided components."""
  try:
    try:
      request_json = json.loads(request_str)
    except json.JSONDecodeError as exp:
      cloud_logging_client.error(
          'Failed to parse request JSON.',
          exp,
      )
      raise pete_errors.InvalidRequestFieldError(
          'Failed to parse request json.'
      ) from exp
    return predictor.predict(request_json, model_runner)
  except pete_errors.PeteError as err:
    return embedding_response.prediction_error_response_v2(
        pete_error_mapping.get_error_code(err)
    )
  except Exception as err:
    cloud_logging_client.error(
        'Unexpected exception raised while processing request.', err
    )
    raise


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  pete_logging.init_application_logging()

  try:
    with pete_predictor_v2.PetePredictor() as predictor:
      model_runner = server_model_runner.ServerModelRunner()

      cloud_logging_client.info('Starting pete prediction executor loop.')
      while True:
        pete_logging.init_embedding_request_logging()
        cloud_logging_client.debug('Waiting for request.')
        try:
          request_str = sys.stdin.readline()
        except EOFError:
          cloud_logging_client.debug('EOF on input, exiting.')
          return
        start_time = time.time()
        cloud_logging_client.debug('Received request.')
        result_json = _run_request(request_str, predictor, model_runner)
        cloud_logging_client.debug('Returning result from executor.')
        try:
          json.dump(result_json, sys.stdout)
          sys.stdout.write('\n')
          sys.stdout.flush()
        except BrokenPipeError:
          cloud_logging_client.debug('Pipe broken, exiting.')
          return
        elapsed = time.time() - start_time
        cloud_logging_client.info(f'Finished handling request ({elapsed} sec).')
  except Exception as exp:
    cloud_logging_client.error('Unhandled exception in executor.', exp)
    raise


if __name__ == '__main__':
  app.run(main)
