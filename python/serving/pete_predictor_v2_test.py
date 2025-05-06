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

"""Unit tests for pathology 2.0 endpoint predictor."""

import base64
import dataclasses
import io
import json
import os
import shutil
import typing
from typing import Any, List, Mapping, Sequence
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import cv2
from ez_wsi_dicomweb import credential_factory
from ez_wsi_dicomweb import dicom_slide
from ez_wsi_dicomweb import dicom_web_interface
from ez_wsi_dicomweb import gcs_image
from ez_wsi_dicomweb import patch_embedding_endpoints
from ez_wsi_dicomweb.ml_toolkit import dicom_path
import google.auth
import jsonschema
import numpy as np
from PIL import ImageCms
import PIL.Image
import pydicom
import yaml

from serving.serving_framework import model_runner
from serving import pete_errors
from serving import pete_icc_profile_cache
from serving import pete_predictor_v2
from serving.data_models import embedding_converter
from serving.data_models import embedding_request
from serving.data_models import patch_coordinate
from serving.test_utils import test_files
from ez_wsi_dicomweb.test_utils.dicom_store_mock import dicom_store_mock
from ez_wsi_dicomweb.test_utils.gcs_mock import gcs_mock


_EndpointJsonKeys = patch_embedding_endpoints.EndpointJsonKeys

_OPEN_API_RESPONSE_YAML_PATH = os.path.join(
    os.path.dirname(__file__), 'vertex_schemata', 'prediction.yaml')

_PETE_REQUEST = {
    _EndpointJsonKeys.INSTANCES: [
        {
            _EndpointJsonKeys.DICOM_PATH: {
                _EndpointJsonKeys.SERIES_PATH: None,  # Will be defined later
                _EndpointJsonKeys.INSTANCE_UIDS: None,  # Will be defined later
            },
            _EndpointJsonKeys.EXTENSIONS: {
                pete_predictor_v2._EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: (
                    pete_predictor_v2._FALSE
                )
            },
            _EndpointJsonKeys.PATCH_COORDINATES: [
                {_EndpointJsonKeys.X_ORIGIN: 1, _EndpointJsonKeys.Y_ORIGIN: 2},
                {_EndpointJsonKeys.X_ORIGIN: 3, _EndpointJsonKeys.Y_ORIGIN: 4},
            ],
        },
        {
            _EndpointJsonKeys.IMAGE_FILE_URI: 'gs://test_bucket/google.jpg',
            _EndpointJsonKeys.EXTENSIONS: {
                pete_predictor_v2._EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: (
                    pete_predictor_v2._FALSE
                )
            },
            _EndpointJsonKeys.PATCH_COORDINATES: [
                {_EndpointJsonKeys.X_ORIGIN: 5, _EndpointJsonKeys.Y_ORIGIN: 6},
                {_EndpointJsonKeys.X_ORIGIN: 7, _EndpointJsonKeys.Y_ORIGIN: 8},
            ],
        },
        {
            _EndpointJsonKeys.RAW_IMAGE_BYTES: None,  # Will be filled in later
            _EndpointJsonKeys.EXTENSIONS: {
                pete_predictor_v2._EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: (
                    pete_predictor_v2._FALSE
                )
            },
            _EndpointJsonKeys.PATCH_COORDINATES: [
                {_EndpointJsonKeys.X_ORIGIN: 9, _EndpointJsonKeys.Y_ORIGIN: 10},
                {
                    _EndpointJsonKeys.X_ORIGIN: 11,
                    _EndpointJsonKeys.Y_ORIGIN: 12,
                },
            ],
        },
    ]
}


def _mock_credentials() -> google.auth.credentials.Credentials:
  credential_mock = mock.create_autospec(
      google.auth.credentials.Credentials, instance=True
  )
  type(credential_mock).token = mock.PropertyMock(return_value='mock_token')
  return credential_mock


def _dicom_instance_patch_request(
    series_path: dicom_path.Path,
    sop_instance_uid: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> Mapping[str, Any]:
  return {
      _EndpointJsonKeys.DICOM_PATH: {
          _EndpointJsonKeys.SERIES_PATH: str(series_path),
          _EndpointJsonKeys.INSTANCE_UIDS: [sop_instance_uid],
      },
      _EndpointJsonKeys.EXTENSIONS: {
          pete_predictor_v2._EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: (
              pete_predictor_v2._FALSE
          )
      },
      _EndpointJsonKeys.PATCH_COORDINATES: [
          {
              _EndpointJsonKeys.X_ORIGIN: x1,
              _EndpointJsonKeys.Y_ORIGIN: y1,
          },
          {
              _EndpointJsonKeys.X_ORIGIN: x2,
              _EndpointJsonKeys.Y_ORIGIN: y2,
          },
      ],
  }


def _create_patch_coordinates(
    count: int,
) -> List[patch_coordinate.PatchCoordinate]:
  return [
      patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
      for _ in range(count)
  ]


def _dicom_series_path(dcm: pydicom.Dataset) -> dicom_path.Path:
  return dicom_path.FromString(
      f'/projects/project/locations/location/datasets/dataset/dicomStores/dicomstore/dicomWeb/studies/{dcm.StudyInstanceUID}/series/{dcm.SeriesInstanceUID}'
  )


class MockModelRunner:
  """Mock embedding, return mean for each channel in patch."""

  def batch_model(self, data: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    """Compute and return mock embeddings."""
    return [np.mean(d, axis=(1, 2)) for d in data]


_mock_model_runner = typing.cast(model_runner.ModelRunner, MockModelRunner())


def _load_openapi_spec(
    openapi_yaml_path, schema_component=None
) -> jsonschema.Draft202012Validator:
  """Returns a validator for the OpenAPI spec & optionally a component."""
  try:
    with open(openapi_yaml_path, 'r') as f:
      openapi_spec = yaml.safe_load(f)
      validator = jsonschema.Draft202012Validator(openapi_spec)
      if schema_component:
        return validator.evolve(
            schema=openapi_spec['components'][schema_component]
        )
      else:
        return validator.evolve(schema=openapi_spec)
  except (IOError, yaml.YAMLError) as exp:
    raise ValueError('OpenAPI specification failed to load.') from exp


class PetePredictorV2Test(parameterized.TestCase):

  def test_count_patch_coordinates(self):
    patch_counter = pete_predictor_v2._RequestPatchCountSizeMonitor(
        embedding_request.EmbeddingRequestV2([
            embedding_request.DicomImageV2(
                '', '', {}, [], _create_patch_coordinates(3)
            ),
            embedding_request.GcsImageV2(
                '', '', {}, _create_patch_coordinates(2)
            ),
            embedding_request.EmbeddedImageV2(
                '', {}, _create_patch_coordinates(1)
            ),
        ])
    )
    self.assertEqual(patch_counter.count, 6)

  def test_add_count_patch_coordinates(self):
    patch_counter = pete_predictor_v2._RequestPatchCountSizeMonitor(
        embedding_request.EmbeddingRequestV2([
            embedding_request.DicomImageV2(
                '', '', {}, [], _create_patch_coordinates(3)
            ),
            embedding_request.GcsImageV2(
                '', '', {}, _create_patch_coordinates(2)
            ),
            embedding_request.EmbeddedImageV2(
                '', {}, _create_patch_coordinates(1)
            ),
        ])
    )
    patch_counter.add_patch_count(2)
    self.assertEqual(patch_counter.count, 8)

  def test_add_count_patch_coordinates_raises(self):
    patch_counter = pete_predictor_v2._RequestPatchCountSizeMonitor(
        embedding_request.EmbeddingRequestV2([
            embedding_request.DicomImageV2(
                '', '', {}, [], _create_patch_coordinates(3)
            ),
            embedding_request.GcsImageV2(
                '', '', {}, _create_patch_coordinates(2)
            ),
            embedding_request.EmbeddedImageV2(
                '', {}, _create_patch_coordinates(1)
            ),
        ])
    )
    with self.assertRaises(pete_errors.TooManyPatchesError):
      patch_counter.add_patch_count(999999)

  def test_count_patch_coordinates_invalid_object_raises(self):
    with self.assertRaises(pete_errors.InternalBugError):
      pete_predictor_v2._RequestPatchCountSizeMonitor(
          embedding_request.EmbeddingRequestV2([
              embedding_request.DicomImageV2(
                  '', '', {}, [], _create_patch_coordinates(3)
              ),
              embedding_request.GcsImageV2(
                  '', '', {}, _create_patch_coordinates(2)
              ),
              'str',
              embedding_request.EmbeddedImageV2(
                  '', {}, _create_patch_coordinates(1)
              ),
          ])
      )

  def test_pete_predictor_not_run_in_context_managed_block_raises(self):
    with self.assertRaisesRegex(
        pete_errors.InternalBugError,
        'PetePredictor must be run in context mangaged block.',
    ):
      pete_predictor_v2.PetePredictor().predict({}, _mock_model_runner)

  def test_pete_predictor_dicom_exceed_max_embeddings_raises(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    json_metadata = {
        _EndpointJsonKeys.INSTANCES: [
            dict(
                dicom_path=dict(
                    series_path=str(path),
                    instance_uids=[dcm.SOPInstanceUID],
                ),
                patch_coordinates=[dict(x_origin=0, y_origin=0)],
            ),
        ] * (pete_predictor_v2._MAX_PATCHES_PER_REQUEST + 1)
    }
    with self.assertRaises(pete_errors.TooManyPatchesError):
      with pete_predictor_v2.PetePredictor() as predictor:
        predictor.predict(json_metadata, _mock_model_runner)

  def test_get_gcs_jpeg_image_patches(self):
    tdir = self.create_tempdir()
    shutil.copyfile(
        test_files.testdata_path('google.jpg'), os.path.join(tdir, 'google.jpg')
    )
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
      ]
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/google.jpg',
          '',
          {_EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False},
          coordinates,
      )
      patches = pete_predictor_v2._get_gcs_patches(ei)
      self.assertEqual(patches.shape, (1, 224, 224, 3))
      self.assertEqual(np.min(patches), 0)
      self.assertEqual(np.max(patches), 1.0)

  @parameterized.parameters(['', 'mock_bearer_token'])
  def test_get_gcs_jpeg_image_patche_bytes(self, bearer_token):
    tdir = self.create_tempdir()
    shutil.copyfile(
        test_files.testdata_path('google.jpg'), os.path.join(tdir, 'google.jpg')
    )
    with PIL.Image.open(test_files.testdata_path('google.jpg')) as img:
      expected_bytes = np.asarray(img)
      # Image is smaller than patch pad with zeros.
      expected_bytes = np.pad(
          expected_bytes,
          ((0, 68), (0, 0), (0, 0)),
          'constant',
          constant_values=0,
      )
      expected_bytes = expected_bytes.astype(np.float32) / 255.0
    expected_bytes = np.expand_dims(expected_bytes[:224, :224, :], axis=0)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/google.jpg',
          bearer_token,
          {_EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False},
          coordinates,
      )
      patches = pete_predictor_v2._get_gcs_patches(ei)
    np.testing.assert_array_almost_equal(patches, expected_bytes)

  @parameterized.named_parameters([
      dict(testcase_name='bad_blob', bad_path='gs://test_bucket/not_found.jpg'),
      dict(testcase_name='bad_bucket', bad_path='gs://bad_bucket/google.jpg'),
  ])
  def test_get_gcs_image_patches_not_found(self, bad_path):
    tdir = self.create_tempdir()
    shutil.copyfile(
        test_files.testdata_path('google.jpg'), os.path.join(tdir, 'google.jpg')
    )
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
      ]
      ei = embedding_request.GcsImageV2(bad_path, '', {}, coordinates)
      with self.assertRaises((
          pete_errors.HttpError,
          pete_errors.InvalidCredentialsError,
          pete_errors.ImageError,
      )):
        pete_predictor_v2._get_gcs_patches(ei)

  @parameterized.named_parameters([
      dict(testcase_name='invalid_path', bad_path='garbage'),
      dict(testcase_name='empty', bad_path=''),
  ])
  def test_get_gcs_image_patches_path_format_error_raises(self, bad_path):
    tdir = self.create_tempdir()
    shutil.copyfile(
        test_files.testdata_path('google.jpg'), os.path.join(tdir, 'google.jpg')
    )
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
      ]
      ei = embedding_request.GcsImageV2(bad_path, '', {}, coordinates)
      with self.assertRaises(pete_errors.GcsImagePathFormatError):
        pete_predictor_v2._get_gcs_patches(ei)

  def test_get_gcs_image_patches_bad_bytes(
      self,
  ):
    tdir = self.create_tempdir()
    with open(os.path.join(tdir, 'garbage.jpg'), 'wb') as outfile:
      outfile.write(b'badf00d')
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
      ]
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/garbage.jpg', '', {}, coordinates
      )
      with self.assertRaises(pete_errors.ImageError):
        pete_predictor_v2._get_gcs_patches(ei)

  def test_get_gcs_png_image_patches(self):
    tdir = self.create_tempdir()
    with PIL.Image.open(test_files.testdata_path('google.jpg')) as img:
      img.save(os.path.join(tdir, 'google.png'), 'png')
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
      ]
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/google.png',
          '',
          {_EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False},
          coordinates,
      )
      patches = pete_predictor_v2._get_gcs_patches(ei)
    self.assertEqual(patches.shape, (1, 224, 224, 3))
    self.assertEqual(np.min(patches), 0)
    self.assertEqual(np.max(patches), 1.0)

  @mock.patch.object(
      google.auth,
      'default',
      return_value=(_mock_credentials(), 'mock_project'),
  )
  def test_get_gcs_patches_from_patch_level_metadata(self, unused_mock):
    tdir = self.create_tempdir()
    shutil.copyfile(
        test_files.testdata_path('google.jpg'), os.path.join(tdir, 'google.jpg')
    )
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      image = gcs_image.GcsImage('gs://test_bucket/google.jpg')
      patch_1 = image.get_patch(0, 0, 224, 224)
      patch_2 = image.get_patch(10, 0, 224, 224)
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0),
          patch_coordinate.create_patch_coordinate(x_origin=10, y_origin=0),
      ]
      state = {'patches': [patch_1.json_metadata(), patch_2.json_metadata()]}
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/google.jpg',
          '',
          {
              _EndpointJsonKeys.EZ_WSI_STATE: state,
              _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: 'False',
          },
          coordinates,
      )
    patches = pete_predictor_v2._get_gcs_patches(ei)
    self.assertEqual(patches.shape, (2, 224, 224, 3))
    self.assertEqual(np.min(patches), 0)
    self.assertEqual(np.max(patches), 1.0)

  @mock.patch.object(
      google.auth,
      'default',
      return_value=(_mock_credentials(), 'mock_project'),
  )
  def test_get_gcs_patches_metadata_and_request_count_not_match_raise(
      self, unused_mock
  ):
    tdir = self.create_tempdir()
    shutil.copyfile(
        test_files.testdata_path('google.jpg'), os.path.join(tdir, 'google.jpg')
    )
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      image = gcs_image.GcsImage('gs://test_bucket/google.jpg')
      patch_1 = image.get_patch(0, 0, 224, 224)
      patch_2 = image.get_patch(10, 0, 224, 224)
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0),
      ]
      state = {'patches': [patch_1.json_metadata(), patch_2.json_metadata()]}
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/google.jpg',
          '',
          {
              _EndpointJsonKeys.EZ_WSI_STATE: state,
              _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: 'False',
          },
          coordinates,
      )
    with self.assertRaisesRegex(
        pete_errors.EzWsiStateError,
        'Length of per-patch and patch coordinates data do not match.',
    ):
      pete_predictor_v2._get_gcs_patches(ei)

  @mock.patch.object(
      google.auth,
      'default',
      return_value=(_mock_credentials(), 'mock_project'),
  )
  def test_get_gcs_patches_from_image_level_metadata(self, unused_mock):
    tdir = self.create_tempdir()
    shutil.copyfile(
        test_files.testdata_path('google.jpg'), os.path.join(tdir, 'google.jpg')
    )
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      image = gcs_image.GcsImage('gs://test_bucket/google.jpg')
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0),
          patch_coordinate.create_patch_coordinate(x_origin=10, y_origin=0),
      ]
      state = {'image': image.json_metadata()}
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/google.jpg',
          '',
          {
              _EndpointJsonKeys.EZ_WSI_STATE: state,
              _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: 'False',
          },
          coordinates,
      )
    patches = pete_predictor_v2._get_gcs_patches(ei)
    self.assertEqual(patches.shape, (2, 224, 224, 3))
    self.assertEqual(np.min(patches), 0)
    self.assertEqual(np.max(patches), 1.0)

  @mock.patch.object(
      google.auth,
      'default',
      return_value=(_mock_credentials(), 'mock_project'),
  )
  def test_get_gcs_patches_from_image_level_bad_metadata_raises(
      self, unused_mock
  ):
    tdir = self.create_tempdir()
    shutil.copyfile(
        test_files.testdata_path('google.jpg'), os.path.join(tdir, 'google.jpg')
    )
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0),
          patch_coordinate.create_patch_coordinate(x_origin=10, y_origin=0),
      ]
      state = {'image': 'badf00d'}
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/google.jpg',
          '',
          {
              _EndpointJsonKeys.EZ_WSI_STATE: state,
              _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: 'False',
          },
          coordinates,
      )
    with self.assertRaisesRegex(
        pete_errors.ImageError, 'Invalid image JSON metadata.'
    ):
      pete_predictor_v2._get_gcs_patches(ei)

  @mock.patch.object(
      google.auth,
      'default',
      return_value=(_mock_credentials(), 'mock_project'),
  )
  def test_get_gcs_patches_from_image_level_patch_does_not_fall_in_image_raises(
      self, unused_mock
  ):
    tdir = self.create_tempdir()
    shutil.copyfile(
        test_files.testdata_path('google.jpg'), os.path.join(tdir, 'google.jpg')
    )
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      image = gcs_image.GcsImage('gs://test_bucket/google.jpg')
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0),
          patch_coordinate.create_patch_coordinate(x_origin=10, y_origin=0),
      ]
      state = {'image': image.json_metadata()}
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/google.jpg',
          '',
          {
              _EndpointJsonKeys.EZ_WSI_STATE: state,
              _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: 'True',
          },
          coordinates,
      )
    with self.assertRaisesRegex(
        pete_errors.PatchOutsideOfImageDimensionsError,
        'Patch dimensions .* fall outside of image dimensions .*',
    ):
      pete_predictor_v2._get_gcs_patches(ei)

  @mock.patch.object(
      google.auth,
      'default',
      return_value=(_mock_credentials(), 'mock_project'),
  )
  def test_get_gcs_patches_from_image_patch_does_not_fall_in_image_raises(
      self, unused_mock
  ):
    tdir = self.create_tempdir()
    shutil.copyfile(
        test_files.testdata_path('google.jpg'), os.path.join(tdir, 'google.jpg')
    )
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0),
          patch_coordinate.create_patch_coordinate(x_origin=10, y_origin=0),
      ]
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/google.jpg',
          '',
          {
              _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: 'True',
          },
          coordinates,
      )
      with self.assertRaisesRegex(
          pete_errors.PatchOutsideOfImageDimensionsError,
          'Patch dimensions .* fall outside of image dimensions .*',
      ):
        pete_predictor_v2._get_gcs_patches(ei)

  @mock.patch.object(
      google.auth,
      'default',
      return_value=(_mock_credentials(), 'mock_project'),
  )
  def test_get_gcs_patches_from_invalid_patch_level_metadata_raises(
      self, unused_mock
  ):
    tdir = self.create_tempdir()
    shutil.copyfile(
        test_files.testdata_path('google.jpg'), os.path.join(tdir, 'google.jpg')
    )
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      image = gcs_image.GcsImage('gs://test_bucket/google.jpg')
      patch_1 = image.get_patch(0, 0, 224, 224)
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0),
          patch_coordinate.create_patch_coordinate(x_origin=10, y_origin=0),
      ]
      state = {'patches': [patch_1.json_metadata(), 'badfood']}
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/google.jpg',
          '',
          {
              _EndpointJsonKeys.EZ_WSI_STATE: state,
              _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: 'False',
          },
          coordinates,
      )
    with self.assertRaisesRegex(
        pete_errors.ImageError,
        'Error occurred reading image; Error decoding image bytes.',
    ):
      pete_predictor_v2._get_gcs_patches(ei)

  def test_get_embedded_jpeg_image_patches(self):
    with open(test_files.testdata_path('google.jpg'), 'rb') as imbytes:
      encoding = base64.b64encode(imbytes.read())
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    ei = embedding_request.EmbeddedImageV2(
        encoding,
        {_EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False},
        coordinates,
    )
    patches = pete_predictor_v2._get_embedded_image_patches(ei)
    self.assertEqual(patches.shape, (1, 224, 224, 3))
    self.assertEqual(np.min(patches), 0)
    self.assertEqual(np.max(patches), 1.0)

  def test_get_embedded_jpeg_image_patche_bytes(self):
    with open(test_files.testdata_path('google.jpg'), 'rb') as imbytes:
      encoding = base64.b64encode(imbytes.read())
    with PIL.Image.open(test_files.testdata_path('google.jpg')) as img:
      expected_bytes = np.asarray(img)
      # Image is smaller than patch pad with zeros.
      expected_bytes = np.pad(
          expected_bytes,
          ((0, 68), (0, 0), (0, 0)),
          'constant',
          constant_values=0,
      )
      expected_bytes = expected_bytes.astype(np.float32) / 255.0
    expected_bytes = np.expand_dims(expected_bytes[:224, :224, :], axis=0)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    ei = embedding_request.EmbeddedImageV2(
        encoding,
        {_EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False},
        coordinates,
    )
    patches = pete_predictor_v2._get_embedded_image_patches(ei)
    np.testing.assert_array_almost_equal(patches, expected_bytes)

  def test_get_embedded_png_image_patches(self):
    with io.BytesIO() as buff:
      with PIL.Image.open(test_files.testdata_path('google.jpg')) as img:
        img.save(buff, 'png')
      encoding = base64.b64encode(buff.getvalue())
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    ei = embedding_request.EmbeddedImageV2(
        encoding,
        {_EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False},
        coordinates,
    )
    patches = pete_predictor_v2._get_embedded_image_patches(ei)
    self.assertEqual(patches.shape, (1, 224, 224, 3))
    self.assertEqual(np.min(patches), 0)
    self.assertEqual(np.max(patches), 1.0)

  def test_get_embedded_image_bad_bytes(self):
    encoding = base64.b64encode(b'badf00d')
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    ei = embedding_request.EmbeddedImageV2(encoding, {}, coordinates)
    with self.assertRaises(pete_errors.ImageError):
      pete_predictor_v2._get_embedded_image_patches(ei)

  def test_get_dicom_patches_instance_not_found(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    instance = embedding_request.DicomImageV2(
        str(path), 'mock_token', {}, ['1.42'], coordinates
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      with self.assertRaises(pete_errors.DicomPathError):
        pete_predictor_v2._get_dicom_patches(
            instance,
            pete_predictor_v2._RequestPatchCountSizeMonitor(
                embedding_request.EmbeddingRequestV2([])
            ),
        )

  def test_get_dicom_patches_dicom_slide_not_found(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = dicom_path.FromString(
        '/projects/project/locations/location/datasets/dataset/dicomStores/dicomstore/dicomWeb/studies/1.42/series/1.42'
    )
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    instance = embedding_request.DicomImageV2(
        str(path), 'mock_token', {}, ['1.42'], coordinates
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      with self.assertRaises(pete_errors.DicomPathError):
        pete_predictor_v2._get_dicom_patches(
            instance,
            pete_predictor_v2._RequestPatchCountSizeMonitor(
                embedding_request.EmbeddingRequestV2([])
            ),
        )

  @parameterized.parameters(['https://bad_path', '', 'bad_path'])
  def test_get_dicom_patches_bad_path(self, bad_path):
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    instance = embedding_request.DicomImageV2(
        bad_path, 'mock_token', {}, ['1.42'], coordinates
    )
    with self.assertRaises(pete_errors.DicomPathError):
      pete_predictor_v2._get_dicom_patches(
          instance,
          pete_predictor_v2._RequestPatchCountSizeMonitor(
              embedding_request.EmbeddingRequestV2([])
          ),
      )

  @parameterized.parameters(['', 'mock_bearer_token'])
  def test_get_dicom_patches(self, bearer_token):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    instance = embedding_request.DicomImageV2(
        str(path), bearer_token, {}, [dcm.SOPInstanceUID], coordinates
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      patches = pete_predictor_v2._get_dicom_patches(
          instance,
          pete_predictor_v2._RequestPatchCountSizeMonitor(
              embedding_request.EmbeddingRequestV2([])
          ),
      )
    self.assertEqual(patches.shape, (1, 224, 224, 3))
    self.assertEqual(round(float(np.min(patches)), 4), 0.1059)
    self.assertEqual(float(np.max(patches)), 1.0)

  @parameterized.named_parameters([
      dict(testcase_name='no_extension', extension={}),
      dict(
          testcase_name='defines_icc_profile_transform',
          extension={
              patch_embedding_endpoints.EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: (
                  'SRGB'
              )
          },
      ),
  ])
  @mock.patch.object(
      dicom_slide, 'create_icc_profile_transformation', autospec=True
  )
  def test_get_patches_from_dicom_with_out_icc_profile_not_create_transform(
      self, create_transform, extension
  ):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    instance = embedding_request.DicomImageV2(
        str(path),
        'mock_bearer_token',
        extension,
        [dcm.SOPInstanceUID],
        coordinates,
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      patches = pete_predictor_v2._get_dicom_patches(
          instance,
          pete_predictor_v2._RequestPatchCountSizeMonitor(
              embedding_request.EmbeddingRequestV2([])
          ),
      )
    self.assertEqual(patches.shape, (1, 224, 224, 3))
    self.assertEqual(round(float(np.min(patches)), 4), 0.1059)
    self.assertEqual(float(np.max(patches)), 1.0)
    create_transform.assert_not_called()

  def test_get_dicom_patches_no_pixel_spacing(self):
    dcm = pydicom.dcmread(test_files.testdata_path('test.dcm'))
    # remove pixel spacing
    del dcm['SharedFunctionalGroupsSequence']
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    instance = embedding_request.DicomImageV2(
        str(path),
        'mock_bearer_token',
        {_EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False},
        [dcm.SOPInstanceUID],
        coordinates,
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      patches = pete_predictor_v2._get_dicom_patches(
          instance,
          pete_predictor_v2._RequestPatchCountSizeMonitor(
              embedding_request.EmbeddingRequestV2([])
          ),
      )
    self.assertEqual(patches.shape, (1, 224, 224, 3))
    self.assertEqual(round(float(np.min(patches)), 4), 0.0)
    self.assertEqual(float(np.max(patches)), 1.0)

  @parameterized.named_parameters([
      dict(
          testcase_name='VL_SLIDE_COORDINATES_MIROSCOPIC_IMAGE_SOP_CLASS_UID',
          sop_class_uid='1.2.840.10008.5.1.4.1.1.77.1.3',
      ),
      dict(
          testcase_name='VL_MIROSCOPIC_IMAGE_SOP_CLASS_UID ',
          sop_class_uid='1.2.840.10008.5.1.4.1.1.77.1.2',
      ),
  ])
  def test_get_dicom_patches_from_non_tiled_dicom(self, sop_class_uid):
    dcm = pydicom.dcmread(test_files.testdata_path('test.dcm'))
    # remove pixel spacing
    del dcm['SharedFunctionalGroupsSequence']
    dcm.file_meta.MediaStorageSOPClassUID = sop_class_uid
    dcm.SOPClassUID = sop_class_uid
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    instance = embedding_request.DicomImageV2(
        str(path),
        'mock_bearer_token',
        {_EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: False},
        [dcm.SOPInstanceUID],
        coordinates,
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      patches = pete_predictor_v2._get_dicom_patches(
          instance,
          pete_predictor_v2._RequestPatchCountSizeMonitor(
              embedding_request.EmbeddingRequestV2([])
          ),
      )
    self.assertEqual(patches.shape, (1, 224, 224, 3))
    self.assertEqual(round(float(np.min(patches)), 4), 0.0)
    self.assertEqual(float(np.max(patches)), 1.0)

  def test_get_dicom_patches_from_sparse_dicom_raises(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    del dcm['00209311']
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    instance = embedding_request.DicomImageV2(
        str(path), 'bearer_token', {}, [dcm.SOPInstanceUID], coordinates
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      with self.assertRaises(pete_errors.DicomTiledFullError):
        pete_predictor_v2._get_dicom_patches(
            instance,
            pete_predictor_v2._RequestPatchCountSizeMonitor(
                embedding_request.EmbeddingRequestV2([])
            ),
        )

  def test_get_dicom_patches_from_invalid_concatenation_raises(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    dcm_2 = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    dcm_2.SopInstanceUID = '1.42'
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    instance = embedding_request.DicomImageV2(
        str(path),
        'bearer_token',
        {},
        [dcm.SOPInstanceUID, dcm_2.SOPInstanceUID],
        coordinates,
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      mk_dicom_stores[store_path].add_instance(dcm_2)
      with self.assertRaises(pete_errors.InstancesNotConcatenatedError):
        pete_predictor_v2._get_dicom_patches(
            instance,
            pete_predictor_v2._RequestPatchCountSizeMonitor(
                embedding_request.EmbeddingRequestV2([])
            ),
        )

  def test_get_dicom_patches_from_tiled_and_non_tiled_in_same_request_raises(
      self,
  ):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    dcm_2 = pydicom.dcmread(test_files.testdata_path('test.dcm'))
    dcm_2.StudyInstanceUID = dcm.StudyInstanceUID
    dcm_2.SeriesInstanceUID = dcm.SeriesInstanceUID
    dcm_2.file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.3'
    dcm_2.SOPClassUID = '1.2.840.10008.5.1.4.1.1.77.1.3'

    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    instance = embedding_request.DicomImageV2(
        str(path),
        'bearer_token',
        {},
        [dcm.SOPInstanceUID, dcm_2.SOPInstanceUID],
        coordinates,
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      mk_dicom_stores[store_path].add_instance(dcm_2)
      with self.assertRaises(pete_errors.DicomError):
        pete_predictor_v2._get_dicom_patches(
            instance,
            pete_predictor_v2._RequestPatchCountSizeMonitor(
                embedding_request.EmbeddingRequestV2([])
            ),
        )

  def test_get_dicom_patches_from_missing_instance_raises(
      self,
  ):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    instance = embedding_request.DicomImageV2(
        str(path),
        'bearer_token',
        {},
        ['1.42'],
        coordinates,
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      with self.assertRaises(pete_errors.DicomPathError):
        pete_predictor_v2._get_dicom_patches(
            instance,
            pete_predictor_v2._RequestPatchCountSizeMonitor(
                embedding_request.EmbeddingRequestV2([])
            ),
        )

  @parameterized.parameters('mock_bearer_token', '')
  def test_repeated_get_dicom_patches_does_not_re_int(self, bearer_token):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    instance = embedding_request.DicomImageV2(
        str(path), bearer_token, {}, [dcm.SOPInstanceUID], coordinates
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)

      _ = pete_predictor_v2._get_dicom_patches(
          instance,
          pete_predictor_v2._RequestPatchCountSizeMonitor(
              embedding_request.EmbeddingRequestV2([])
          ),
      )
      # repeate prior call
      patches = pete_predictor_v2._get_dicom_patches(
          instance,
          pete_predictor_v2._RequestPatchCountSizeMonitor(
              embedding_request.EmbeddingRequestV2([])
          ),
      )
      # check patch results are as expected.
      self.assertEqual(patches.shape, (1, 224, 224, 3))
      self.assertEqual(round(float(np.min(patches)), 4), 0.1059)
      self.assertEqual(float(np.max(patches)), 1.0)

  def test_pete_predictor_dicom(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    tdir = self.create_tempdir()
    shutil.copyfile(
        test_files.testdata_path('google.jpg'), os.path.join(tdir, 'google.jpg')
    )
    with open(test_files.testdata_path('google.jpg'), 'rb') as imbytes:
      image_bytes = base64.b64encode(imbytes.read())

    json_metadata = {
        _EndpointJsonKeys.INSTANCES: [
            dict(
                dicom_path=dict(
                    series_path=str(path), instance_uids=[dcm.SOPInstanceUID]
                ),
                extensions={
                    _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: (
                        pete_predictor_v2._FALSE
                    )
                },
                patch_coordinates=[dict(x_origin=0, y_origin=0)],
            ),
            dict(
                image_file_uri='gs://test_bucket/google.jpg',
                extensions={
                    _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: (
                        pete_predictor_v2._FALSE
                    )
                },
                patch_coordinates=[dict(x_origin=0, y_origin=0)],
            ),
            {
                _EndpointJsonKeys.RAW_IMAGE_BYTES: image_bytes.decode('utf-8'),
                _EndpointJsonKeys.EXTENSIONS: {
                    _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: (
                        pete_predictor_v2._FALSE
                    )
                },
                _EndpointJsonKeys.PATCH_COORDINATES: [
                    dict(x_origin=0, y_origin=0)
                ],
            },
        ]
    }
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      store_path = str(path.GetStorePath())
      with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
        mk_dicom_stores[store_path].add_instance(dcm)
        with pete_predictor_v2.PetePredictor() as predictor:
          results = predictor.predict(json_metadata, _mock_model_runner)
    self.assertEqual(
        results,
        {
            _EndpointJsonKeys.PREDICTIONS: [
                {
                    _EndpointJsonKeys.RESULT: {
                        _EndpointJsonKeys.PATCH_EMBEDDINGS: [{
                            _EndpointJsonKeys.EMBEDDING_VECTOR: [
                                0.7753128409385681,
                                0.7156112194061279,
                                0.8265919089317322,
                            ],
                            _EndpointJsonKeys.PATCH_COORDINATE: {
                                _EndpointJsonKeys.X_ORIGIN: 0,
                                _EndpointJsonKeys.Y_ORIGIN: 0,
                                _EndpointJsonKeys.WIDTH: 224,
                                _EndpointJsonKeys.HEIGHT: 224,
                            },
                        }]
                    },
                },
                {
                    _EndpointJsonKeys.RESULT: {
                        _EndpointJsonKeys.PATCH_EMBEDDINGS: [{
                            _EndpointJsonKeys.EMBEDDING_VECTOR: [
                                0.6209772229194641,
                                0.6062123775482178,
                                0.6281186938285828,
                            ],
                            _EndpointJsonKeys.PATCH_COORDINATE: {
                                _EndpointJsonKeys.X_ORIGIN: 0,
                                _EndpointJsonKeys.Y_ORIGIN: 0,
                                _EndpointJsonKeys.WIDTH: 224,
                                _EndpointJsonKeys.HEIGHT: 224,
                            },
                        }]
                    },
                },
                {
                    _EndpointJsonKeys.RESULT: {
                        _EndpointJsonKeys.PATCH_EMBEDDINGS: [{
                            _EndpointJsonKeys.EMBEDDING_VECTOR: [
                                0.6209772229194641,
                                0.6062123775482178,
                                0.6281186938285828,
                            ],
                            _EndpointJsonKeys.PATCH_COORDINATE: {
                                _EndpointJsonKeys.X_ORIGIN: 0,
                                _EndpointJsonKeys.Y_ORIGIN: 0,
                                _EndpointJsonKeys.WIDTH: 224,
                                _EndpointJsonKeys.HEIGHT: 224,
                            },
                        }]
                    },
                },
            ]
        },
    )

  def test_get_required_fully_in_source_image_extension_default(self):
    self.assertTrue(
        pete_predictor_v2._get_required_fully_in_source_image_extension({}),
    )

  @parameterized.parameters([True, 'TRUE', 'True', 'true', 'TrUe'])
  def test_get_required_fully_in_source_image_extension_true(self, val):
    self.assertTrue(
        pete_predictor_v2._get_required_fully_in_source_image_extension(
            {_EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: val}
        ),
    )

  @parameterized.parameters([False, 'FALSE', 'False', 'false', 'FaLsE'])
  def test_get_required_fully_in_source_image_extension_false(self, val):
    self.assertFalse(
        pete_predictor_v2._get_required_fully_in_source_image_extension(
            {_EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: val}
        ),
    )

  @parameterized.parameters(['', 'BadString', 1, 2, ({},), ([],), 0, 0.0])
  def test_get_required_fully_in_source_image_extension_raises(self, val):
    with self.assertRaises(pete_errors.InvalidRequestFieldError):
      pete_predictor_v2._get_required_fully_in_source_image_extension(
          {_EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: val}
      )

  def test_get_resize_image_dimensions_succeeds(self):
    val = {
        _EndpointJsonKeys.IMAGE_DIMENSIONS: {'width_px': 10, 'height_px': 20}
    }
    test = pete_predictor_v2._get_resize_image_dimensions(val)
    self.assertEqual((test.width_px, test.height_px), (10, 20))

  def test_get_resize_image_dimensions_returns_none(self):
    self.assertIsNone(pete_predictor_v2._get_resize_image_dimensions({}))

  @parameterized.named_parameters([
      dict(
          testcase_name='dict_ref_int',
          val={_EndpointJsonKeys.IMAGE_DIMENSIONS: 1},
      ),
      dict(
          testcase_name='dict_ref_list',
          val={_EndpointJsonKeys.IMAGE_DIMENSIONS: [1, 2, 3]},
      ),
      dict(
          testcase_name='dict_ref_string_invalid_json',
          val={_EndpointJsonKeys.IMAGE_DIMENSIONS: '1,2,3'},
      ),
      dict(
          testcase_name='dict_ref_string_missing_value',
          val={_EndpointJsonKeys.IMAGE_DIMENSIONS: '{}'},
      ),
      dict(
          testcase_name='dict_ref_string_json_list',
          val={_EndpointJsonKeys.IMAGE_DIMENSIONS: '[1,2,3]'},
      ),
      dict(
          testcase_name='dict_ref_width_only',
          val={_EndpointJsonKeys.IMAGE_DIMENSIONS: {'width_px': 1}},
      ),
      dict(
          testcase_name='dict_ref_height_only',
          val={_EndpointJsonKeys.IMAGE_DIMENSIONS: {'height_px': 1}},
      ),
      dict(
          testcase_name='dict_ref_width_zero',
          val={
              _EndpointJsonKeys.IMAGE_DIMENSIONS: {
                  'width_px': 0,
                  'height_px': 1,
              }
          },
      ),
      dict(
          testcase_name='dict_ref_height_zero',
          val={
              _EndpointJsonKeys.IMAGE_DIMENSIONS: {
                  'width_px': 1,
                  'height_px': 0,
              }
          },
      ),
      dict(
          testcase_name='dict_ref_width_minus_one',
          val={
              _EndpointJsonKeys.IMAGE_DIMENSIONS: {
                  'width_px': -1,
                  'height_px': 1,
              }
          },
      ),
      dict(
          testcase_name='dict_ref_height_minus_one',
          val={
              _EndpointJsonKeys.IMAGE_DIMENSIONS: {
                  'width_px': 1,
                  'height_px': -1,
              }
          },
      ),
  ])
  def test_get_resize_image_dimensions_raises(self, val):
    with self.assertRaises(pete_errors.InvalidRequestFieldError):
      pete_predictor_v2._get_resize_image_dimensions(val)

  @parameterized.parameters([(10, 20), (20, 10), (20, 20)])
  def test_get_resize_image_dimensions_raises_if_exceeds_max(
      self, width, height
  ):
    val = {
        _EndpointJsonKeys.IMAGE_DIMENSIONS: (
            '{"width_px": %d, "height_px": %d}' % (width, height)
        )
    }
    with self.assertRaises(pete_errors.InvalidRequestFieldError):
      pete_predictor_v2._get_resize_image_dimensions(val, 15)

  def test_get_resize_image_dimensions_does_not_raises_if_at_max(self):
    val = {
        _EndpointJsonKeys.IMAGE_DIMENSIONS: {'width_px': 15, 'height_px': 10}
    }
    dim = pete_predictor_v2._get_resize_image_dimensions(val, 15)
    self.assertEqual((dim.width_px, dim.height_px), (15, 10))

  @parameterized.parameters([1, 1.2, 'abc', ([],)])
  def test_get_ez_wsi_state_invalid_value_raises(self, val):
    with self.assertRaises(pete_errors.EzWsiStateError):
      pete_predictor_v2._get_ez_wsi_state({_EndpointJsonKeys.EZ_WSI_STATE: val})

  def test_get_ez_wsi_state_default(self):
    self.assertEqual(pete_predictor_v2._get_ez_wsi_state({}), {})

  def test_get_ez_wsi_state_expected(self):
    expected = {'abc': 123}
    self.assertEqual(
        pete_predictor_v2._get_ez_wsi_state(
            {_EndpointJsonKeys.EZ_WSI_STATE: expected}
        ),
        expected,
    )

  def test_get_dicom_instances_with_different_transfer_syntax_raise(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    # define two concatenation instances with different transfer syntaxs.
    dcm.InConcatenationNumber = 1
    dcm.ConcatenationUID = '1.43'
    dcm.ConcatenationFrameOffsetNumber = 0
    dcm2 = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    dcm2.file_meta.MediaStorageSOPInstanceUID = '1.42'
    dcm2.ConcatenationFrameOffsetNumber = dcm.NumberOfFrames
    dcm2.SOPInstanceUID = '1.42'
    dcm2.InConcatenationNumber = 2
    dcm2.ConcatenationUID = '1.43'
    dcm2.file_meta.TransferSyntaxUID = '1.2.840.10008.1.​2.​1'
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    instance = embedding_request.DicomImageV2(
        str(path),
        'mock_token',
        {},
        [dcm.SOPInstanceUID, dcm2.SOPInstanceUID],
        coordinates,
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      mk_dicom_stores[store_path].add_instance(dcm2)
      with self.assertRaisesRegex(
          pete_errors.DicomError,
          'All DICOM instances in a pyramid level are required to have same'
          ' TransferSyntaxUID.',
      ):
        pete_predictor_v2._get_dicom_patches(
            instance,
            pete_predictor_v2._RequestPatchCountSizeMonitor(
                embedding_request.EmbeddingRequestV2([])
            ),
        )

  def test_get_dicom_instances_invalid_tags_raises(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    del dcm['00080008']
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    instance = embedding_request.DicomImageV2(
        str(path), 'mock_token', {}, [dcm.SOPInstanceUID], coordinates
    )
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      with self.assertRaisesRegex(
          pete_errors.DicomError, 'DICOM instance missing required tags.'
      ):
        pete_predictor_v2._get_dicom_patches(
            instance,
            pete_predictor_v2._RequestPatchCountSizeMonitor(
                embedding_request.EmbeddingRequestV2([])
            ),
        )

  def test_can_not_find_dicom_level_raises(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      ds = dicom_slide.DicomSlide(
          dicom_web_interface.DicomWebInterface(
              credential_factory.NoAuthCredentialsFactory()
          ),
          path,
      )
      # error occures due to instance requesting predictions for an instance
      # which is not defined in the metadata.
      metadata = ds.json_metadata()
      metadata = metadata.replace(dcm.SOPInstanceUID, '1.42')
      metadata = json.loads(metadata)
      # modifying metadata to remove instance.
      instance = embedding_request.DicomImageV2(
          str(path),
          'mock_token',
          {_EndpointJsonKeys.EZ_WSI_STATE: metadata},
          [dcm.SOPInstanceUID],
          coordinates,
      )
      with self.assertRaises(pete_errors.LevelNotFoundError):
        pete_predictor_v2._get_dicom_patches(
            instance,
            pete_predictor_v2._RequestPatchCountSizeMonitor(
                embedding_request.EmbeddingRequestV2([])
            ),
        )

  def test_dicom_level_resize_greater_than_8x_raises(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = embedding_request.DicomImageV2(
          str(path),
          'mock_token',
          {
              _EndpointJsonKeys.IMAGE_DIMENSIONS: dataclasses.asdict(
                  dicom_slide.ImageDimensions(
                      int(dcm.TotalPixelMatrixColumns // 9),
                      int(dcm.TotalPixelMatrixRows // 9),
                  )
              )
          },
          [dcm.SOPInstanceUID],
          coordinates,
      )
      with self.assertRaisesRegex(
          pete_errors.DicomImageDownsamplingTooLargeError,
          'Image downsampling, 9.09091X, exceeds 8x',
      ):
        pete_predictor_v2._get_dicom_patches(
            instance,
            pete_predictor_v2._RequestPatchCountSizeMonitor(
                embedding_request.EmbeddingRequestV2([])
            ),
        )

  def test_dicom_level_resize(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = embedding_request.DicomImageV2(
          str(path),
          'mock_token',
          {
              _EndpointJsonKeys.IMAGE_DIMENSIONS: dataclasses.asdict(
                  dicom_slide.ImageDimensions(
                      int(dcm.TotalPixelMatrixColumns // 3),
                      int(dcm.TotalPixelMatrixRows // 3),
                  )
              )
          },
          [dcm.SOPInstanceUID],
          coordinates,
      )
      result = pete_predictor_v2._get_dicom_patches(
          instance,
          pete_predictor_v2._RequestPatchCountSizeMonitor(
              embedding_request.EmbeddingRequestV2([])
          ),
      )

      # check patch results are as expected.
      self.assertEqual(result.shape, (1, 224, 224, 3))
      self.assertLessEqual(round(float(np.min(result)), 4), 0.2)
      self.assertLessEqual(round(float(np.max(result)), 4), 0.95)

  def test_dicom_patch_outside_level_dim(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = embedding_request.DicomImageV2(
          str(path),
          'mock_token',
          {
              _EndpointJsonKeys.IMAGE_DIMENSIONS: dataclasses.asdict(
                  dicom_slide.ImageDimensions(
                      int(dcm.TotalPixelMatrixColumns // 7),
                      int(dcm.TotalPixelMatrixRows // 7),
                  )
              )
          },
          [dcm.SOPInstanceUID],
          coordinates,
      )
      with self.assertRaisesRegex(
          pete_errors.PatchOutsideOfImageDimensionsError,
          'Patch dimensions.*fall outside of DICOM level pyramid imaging'
          ' dimensions.*',
      ):
        pete_predictor_v2._get_dicom_patches(
            instance,
            pete_predictor_v2._RequestPatchCountSizeMonitor(
                embedding_request.EmbeddingRequestV2([])
            ),
        )

  def test_dicom_bits_allocated_not_8_raises(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    dcm.BitsAllocated = 12
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = embedding_request.DicomImageV2(
          str(path),
          'mock_token',
          {},
          [dcm.SOPInstanceUID],
          coordinates,
      )
      with self.assertRaisesRegex(
          pete_errors.DicomError,
          'DICOM contains instances with imaging bits allocated != 8',
      ):
        pete_predictor_v2._get_dicom_patches(
            instance,
            pete_predictor_v2._RequestPatchCountSizeMonitor(
                embedding_request.EmbeddingRequestV2([])
            ),
        )

  @parameterized.parameters([1, ({},), ([],), 1.2])
  def test_get_icc_profile_invalid_json_raises_not_a_string(self, val):
    with self.assertRaisesRegex(
        pete_errors.InvalidRequestFieldError,
        f'{_EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE} value is not a'
        ' string',
    ):
      pete_predictor_v2._get_icc_profile(
          {_EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: val}
      )

  def test_get_icc_profile_set_to_invalid_value_raises_not_a_string(self):
    with self.assertRaisesRegex(
        pete_errors.InvalidIccProfileTransformError,
        f'{_EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE} value is not'
        ' valid; expecting: ADOBERGB, ROMMRGB, SRGB, or NONE.',
    ):
      pete_predictor_v2._get_icc_profile(
          {_EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'bad_value'}
      )

  def test_get_icc_profile_missing_key(self):
    self.assertIsNone(pete_predictor_v2._get_icc_profile({}))

  def test_get_icc_profile_none(self):
    self.assertIsNone(
        pete_predictor_v2._get_icc_profile(
            {_EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'noNe'}
        )
    )

  @parameterized.parameters(['adobeRGB', 'ADOBERGB'])
  def test_get_adobe_iccprofile(self, profile_name):
    profile = pete_predictor_v2._get_icc_profile(
        {_EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: profile_name}
    )
    self.assertEqual(
        profile.tobytes(),
        dicom_slide.get_adobergb_icc_profile_bytes(),
    )

  @parameterized.parameters(['sRGB', 'SRGB'])
  def test_get_srgb_iccprofile(self, profile_name):
    profile = pete_predictor_v2._get_icc_profile(
        {_EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: profile_name}
    )
    self.assertEqual(
        profile.tobytes(),
        dicom_slide.get_srgb_icc_profile_bytes(),
    )

  @parameterized.parameters(['rommRGB', 'ROMMRGB'])
  def test_get_rommrgb_iccprofile(self, profile_name):
    profile = pete_predictor_v2._get_icc_profile(
        {_EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: profile_name}
    )
    self.assertEqual(
        profile.tobytes(), dicom_slide.get_rommrgb_icc_profile_bytes()
    )

  @parameterized.parameters(['', 'SRGB'])
  def test_validate_server_and_client_side_icc_profile_normalization(
      self, client_norm
  ):
    self.assertIsNone(
        pete_predictor_v2._validate_server_and_client_side_icc_profile_normalization(
            'SRGB', None, client_norm
        )
    )

  @parameterized.parameters(['', 'NONE'])
  def test_validate_server_and_client_side_icc_profile_normalization_none(
      self, client_norm
  ):
    self.assertIsNone(
        pete_predictor_v2._validate_server_and_client_side_icc_profile_normalization(
            'NONE', None, client_norm
        )
    )

  def test_validate_server_and_client_side_icc_profile_norm_not_match_default_raise(
      self,
  ):
    client_norm = 'SRGB'
    with self.assertRaisesRegex(
        pete_errors.EzWsiStateError,
        'Error different ICC profile transformations defined by client and'
        ' server.',
    ):
      pete_predictor_v2._validate_server_and_client_side_icc_profile_normalization(
          'NONE', None, client_norm
      )

  def test_validate_server_and_client_side_icc_profile_norm_not_match_raise(
      self,
  ):
    client_norm = 'SRGB'
    with self.assertRaisesRegex(
        pete_errors.EzWsiStateError,
        'Error different ICC profile transformations defined by client and'
        ' server.',
    ):
      pete_predictor_v2._validate_server_and_client_side_icc_profile_normalization(
          'ROMMRGB', None, client_norm
      )

  def test_validate_icc_profile_transform_and_client_side_icc_profile_raise(
      self,
  ):
    client_norm = 'SRGB'
    with self.assertRaisesRegex(
        pete_errors.EzWsiStateError,
        'Duplicate ICC profile transformations defined by client and server.',
    ):
      pete_predictor_v2._validate_server_and_client_side_icc_profile_normalization(
          'SRGB',
          mock.create_autospec(ImageCms.ImageCmsTransform, instance=True),
          client_norm,
      )

  def test_dicom_icc_profile_correction_changes_pixel_values(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    dcm.ICCProfile = dicom_slide.get_rommrgb_icc_profile_bytes()
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = embedding_request.DicomImageV2(
          str(path),
          'mock_token',
          {_EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'SRGB'},
          [dcm.SOPInstanceUID],
          coordinates,
      )
      srgb_result = pete_predictor_v2._get_dicom_patches(
          instance,
          pete_predictor_v2._RequestPatchCountSizeMonitor(
              embedding_request.EmbeddingRequestV2([])
          ),
      )
      instance = embedding_request.DicomImageV2(
          str(path),
          'mock_token',
          {_EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'ROMMRGB'},
          [dcm.SOPInstanceUID],
          coordinates,
      )
      rommrgb_result = pete_predictor_v2._get_dicom_patches(
          instance,
          pete_predictor_v2._RequestPatchCountSizeMonitor(
              embedding_request.EmbeddingRequestV2([])
          ),
      )
      # color normalization changes pixel values
      self.assertGreater(np.max(np.abs(rommrgb_result - srgb_result)), 0)

  def test_dicom_icc_profile_no_effect_of_correction_for_same_profile(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    dcm.ICCProfile = dicom_slide.get_rommrgb_icc_profile_bytes()
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = embedding_request.DicomImageV2(
          str(path),
          'mock_token',
          {},
          [dcm.SOPInstanceUID],
          coordinates,
      )
      none_result = pete_predictor_v2._get_dicom_patches(
          instance,
          pete_predictor_v2._RequestPatchCountSizeMonitor(
              embedding_request.EmbeddingRequestV2([])
          ),
      )
      instance = embedding_request.DicomImageV2(
          str(path),
          'mock_token',
          {_EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'ROMMRGB'},
          [dcm.SOPInstanceUID],
          coordinates,
      )
      rommrgb_result = pete_predictor_v2._get_dicom_patches(
          instance,
          pete_predictor_v2._RequestPatchCountSizeMonitor(
              embedding_request.EmbeddingRequestV2([])
          ),
      )
      # no change in changes pixel values
      self.assertEqual(np.max(np.abs(rommrgb_result - none_result)), 0)

  @parameterized.named_parameters([
      dict(testcase_name='no_profile_transform_defined', exensions={}),
      dict(
          testcase_name='none_transform',
          exensions={
              _EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'NONE'
          },
      ),
  ])
  @mock.patch.object(
      pete_icc_profile_cache, 'get_dicom_icc_profile', autospec=True
  )
  def test_dicom_icc_profile_not_called(self, mock_get_profile, exensions):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    dcm.ICCProfile = dicom_slide.get_rommrgb_icc_profile_bytes()
    path = _dicom_series_path(dcm)
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0)
    ]
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)
      instance = embedding_request.DicomImageV2(
          str(path),
          'mock_token',
          exensions,
          [dcm.SOPInstanceUID],
          coordinates,
      )
      pete_predictor_v2._get_dicom_patches(
          instance,
          pete_predictor_v2._RequestPatchCountSizeMonitor(
              embedding_request.EmbeddingRequestV2([])
          ),
      )
      mock_get_profile.assert_not_called()

  @mock.patch.object(
      google.auth,
      'default',
      return_value=(_mock_credentials(), 'mock_project'),
  )
  def test_get_gcs_patches_from_patch_coordinates_outside_of_image_raises(
      self, unused_mock
  ):
    height, width, _ = cv2.imread(test_files.testdata_path('google.jpg')).shape
    tdir = self.create_tempdir()
    shutil.copyfile(
        test_files.testdata_path('google.jpg'), os.path.join(tdir, 'google.jpg')
    )
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      image = gcs_image.GcsImage('gs://test_bucket/google.jpg')
      patch = image.get_patch(-20, -20, 224, 224)
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=-20, y_origin=-20),
      ]
      state = {
          'patches': [patch.json_metadata()],
          'source_image_width_px': width,
          'source_image_height_px': height,
      }
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/google.jpg',
          '',
          {
              _EndpointJsonKeys.EZ_WSI_STATE: state,
              _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: 'True',
          },
          coordinates,
      )
      with self.assertRaisesRegex(
          pete_errors.PatchOutsideOfImageDimensionsError,
          'Patch falls outside of image dimensions.',
      ):
        pete_predictor_v2._get_gcs_patches(ei)

  @mock.patch.object(
      google.auth,
      'default',
      return_value=(_mock_credentials(), 'mock_project'),
  )
  def test_get_gcs_patches_different_client_server_color_transform_raises(
      self, unused_mock
  ):
    tdir = self.create_tempdir()
    shutil.copyfile(
        test_files.testdata_path('google.jpg'), os.path.join(tdir, 'google.jpg')
    )
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0),
      ]
      state = {'icc_profile_metadata_normalization': 'ROMMRGB'}
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/google.jpg',
          '',
          {
              _EndpointJsonKeys.EZ_WSI_STATE: state,
              _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: 'False',
              _EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'SRGB',
          },
          coordinates,
      )
      with self.assertRaisesRegex(
          pete_errors.EzWsiStateError,
          'Error different ICC profile transformations defined by client and'
          ' server.',
      ):
        pete_predictor_v2._get_gcs_patches(ei)

  def test_get_embedded_patches_different_client_server_color_transform_raises(
      self,
  ):
    with open(test_files.testdata_path('google.jpg'), 'rb') as imbytes:
      encoding = base64.b64encode(imbytes.read())
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0),
    ]
    state = {'icc_profile_metadata_normalization': 'ROMMRGB'}
    ei = embedding_request.EmbeddedImageV2(
        encoding,
        {
            _EndpointJsonKeys.EZ_WSI_STATE: state,
            _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: 'False',
            _EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'SRGB',
        },
        coordinates,
    )
    with self.assertRaisesRegex(
        pete_errors.EzWsiStateError,
        'Error different ICC profile transformations defined by client and'
        ' server.',
    ):
      pete_predictor_v2._get_embedded_image_patches(ei)

  @mock.patch.object(
      google.auth,
      'default',
      return_value=(_mock_credentials(), 'mock_project'),
  )
  def test_get_gcs_patches_duplicate_color_transform_raises(self, unused_mock):
    tdir = self.create_tempdir()
    with PIL.Image.open(test_files.testdata_path('google.jpg')) as img:
      img.save(
          os.path.join(tdir, 'google.png'),
          icc_profile=dicom_slide.get_rommrgb_icc_profile_bytes(),
      )
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0),
      ]
      state = {'icc_profile_metadata_normalization': 'ROMMRGB'}
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/google.png',
          '',
          {
              _EndpointJsonKeys.EZ_WSI_STATE: state,
              _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: 'False',
              _EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'ROMMRGB',
          },
          coordinates,
      )
      with self.assertRaisesRegex(
          pete_errors.EzWsiStateError,
          'Duplicate ICC profile transformations defined by client and server.',
      ):
        pete_predictor_v2._get_gcs_patches(ei)

  @mock.patch.object(
      google.auth,
      'default',
      return_value=(_mock_credentials(), 'mock_project'),
  )
  @mock.patch.object(
      pete_predictor_v2, '_get_normalized_patches', autospec=True
  )
  def test_get_gcs_patches_from_image_with_profile_creates_transform(
      self, mock_get_normalized_patches, unused_mock
  ):
    tdir = self.create_tempdir()
    with PIL.Image.open(test_files.testdata_path('google.jpg')) as img:
      img.save(
          os.path.join(tdir, 'google.png'),
          icc_profile=dicom_slide.get_rommrgb_icc_profile_bytes(),
      )
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0),
      ]
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/google.png',
          '',
          {
              _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: 'False',
              _EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'ROMMRGB',
          },
          coordinates,
      )
      pete_predictor_v2._get_gcs_patches(ei)
    mock_get_normalized_patches.assert_called_once()
    self.assertIsNotNone(mock_get_normalized_patches.call_args[0][1])

  @parameterized.named_parameters([
      dict(
          testcase_name='client_side_transform',
          client_side_transform='ROMMRGB',
      ),
      dict(
          testcase_name='no_client_side_transform',
          client_side_transform='NONE',
      ),
  ])
  @mock.patch.object(
      google.auth,
      'default',
      return_value=(_mock_credentials(), 'mock_project'),
  )
  @mock.patch.object(
      pete_predictor_v2, '_get_normalized_patches', autospec=True
  )
  def test_get_gcs_patches_from_image_with_no_profile_creates_no_transform(
      self, mock_get_normalized_patches, unused_mock, client_side_transform
  ):
    tdir = self.create_tempdir()
    with PIL.Image.open(test_files.testdata_path('google.jpg')) as img:
      img.save(os.path.join(tdir, 'google.png'))
    with gcs_mock.GcsMock({'test_bucket': tdir}):
      coordinates = [
          patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0),
      ]
      state = {'icc_profile_metadata_normalization': client_side_transform}
      ei = embedding_request.GcsImageV2(
          'gs://test_bucket/google.png',
          '',
          {
              _EndpointJsonKeys.EZ_WSI_STATE: state,
              _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: 'False',
              _EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'ROMMRGB',
          },
          coordinates,
      )
      pete_predictor_v2._get_gcs_patches(ei)
    mock_get_normalized_patches.assert_called_once()
    self.assertIsNone(mock_get_normalized_patches.call_args[0][1])

  def test_get_embedded_patches_duplicate_color_transform_raises(self):
    tdir = self.create_tempdir()
    path = os.path.join(tdir, 'google.png')
    with PIL.Image.open(test_files.testdata_path('google.jpg')) as img:
      img.save(path, icc_profile=dicom_slide.get_rommrgb_icc_profile_bytes())
    with open(path, 'rb') as imbytes:
      encoding = base64.b64encode(imbytes.read())
    coordinates = [
        patch_coordinate.create_patch_coordinate(x_origin=0, y_origin=0),
    ]
    state = {'icc_profile_metadata_normalization': 'ROMMRGB'}
    ei = embedding_request.EmbeddedImageV2(
        encoding,
        {
            _EndpointJsonKeys.EZ_WSI_STATE: state,
            _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: 'False',
            _EndpointJsonKeys.TRANSFORM_IMAGING_TO_ICC_PROFILE: 'ROMMRGB',
        },
        coordinates,
    )
    with self.assertRaisesRegex(
        pete_errors.EzWsiStateError,
        'Duplicate ICC profile transformations defined by client and server.',
    ):
      pete_predictor_v2._get_embedded_image_patches(ei)

  @mock.patch.object(
      pete_predictor_v2, '_RequestPatchCountSizeMonitor', autospec=True
  )
  @mock.patch.object(
      embedding_converter.EmbeddingConverterV2,
      'json_to_embedding_request',
      autospec=True,
  )
  def test_pete_predictor_raises_if_instance_passed_unidentified_json(
      self, mk, _
  ):

    @dataclasses.dataclass
    class MockResult:
      instances: List[Mapping[str, Any]]

    bad_json = [{'abc': [1, 2, 3]}]
    mk.return_value = MockResult(['bad_object'])
    with self.assertRaisesRegex(
        pete_errors.InternalBugError, 'Unspported instance type.'
    ):
      with pete_predictor_v2.PetePredictor() as predictor:
        predictor.predict(
            bad_json,
            mock.create_autospec(model_runner.ModelRunner, instance=True),
        )

  def test_invalid_gcs_embedding_pete_result(self):
    json_metadata = {
        _EndpointJsonKeys.INSTANCES: [
            dict(
                image_file_uri='gs://test_bucket/google.jpg',
                extensions={
                    _EndpointJsonKeys.REQUIRE_PATCHES_FULLY_IN_SOURCE_IMAGE: (
                        pete_predictor_v2._FALSE
                    )
                },
                patch_coordinates=[
                    dict(x_origin=5, y_origin=6),
                    dict(x_origin=7, y_origin=8),
                ],
            ),
        ] * 2
    }
    with gcs_mock.GcsMock():
      with pete_predictor_v2.PetePredictor() as predictor:
        results = predictor.predict(json_metadata, _mock_model_runner)

    # GCS Mock simulates two errors, for missing files.  Test if either is set
    # then set value to TESTED to validated the whole message.
    for instance in results[_EndpointJsonKeys.PREDICTIONS]:
      self.assertIn(
          instance[_EndpointJsonKeys.ERROR][_EndpointJsonKeys.ERROR_CODE],
          ('HTTP_ERROR', 'INVALID_CREDENTIALS'),
      )
      instance[_EndpointJsonKeys.ERROR][_EndpointJsonKeys.ERROR_CODE] = 'TESTED'
      del instance[_EndpointJsonKeys.ERROR][
          _EndpointJsonKeys.ERROR_CODE_DESCRIPTION
      ]
    self.assertEqual(
        results,
        {
            _EndpointJsonKeys.PREDICTIONS: [
                {
                    _EndpointJsonKeys.ERROR: {
                        _EndpointJsonKeys.ERROR_CODE: 'TESTED'
                    },
                },
                {
                    _EndpointJsonKeys.ERROR: {
                        _EndpointJsonKeys.ERROR_CODE: 'TESTED'
                    },
                },
            ]
        },
    )

  def test_validate_pete_response(self):
    tdir = self.create_tempdir()
    path = os.path.join(tdir, 'google.jpg')
    # save image to location which will back our GCS mock
    shutil.copy(test_files.testdata_path('google.jpg'), path)
    # read image bytes for embedded image.
    with open(test_files.testdata_path('google.jpg'), 'rb') as imbytes:
      image_bytes = base64.b64encode(imbytes.read()).decode('utf-8')
    # read dicom which that will be referenced.
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      with gcs_mock.GcsMock({'test_bucket': tdir}):
        mk_dicom_stores[store_path].add_instance(dcm)
        # fill in the request
        pete_request = _PETE_REQUEST.copy()
        pete_request[_EndpointJsonKeys.INSTANCES][0][
            _EndpointJsonKeys.DICOM_PATH
        ][_EndpointJsonKeys.SERIES_PATH] = str(path)
        pete_request[_EndpointJsonKeys.INSTANCES][0][
            _EndpointJsonKeys.DICOM_PATH
        ][_EndpointJsonKeys.INSTANCE_UIDS] = [dcm.SOPInstanceUID]
        pete_request[_EndpointJsonKeys.INSTANCES][2][
            _EndpointJsonKeys.RAW_IMAGE_BYTES
        ] = image_bytes

        with pete_predictor_v2.PetePredictor() as predictor:
          results = predictor.predict(pete_request, _mock_model_runner)
    validator = _load_openapi_spec(_OPEN_API_RESPONSE_YAML_PATH)
    for result in results[_EndpointJsonKeys.PREDICTIONS]:
      validator.validate(result)

  def test_pete_predictor_good_bad_good_result(self):
    dcm = pydicom.dcmread(
        test_files.testdata_path('multiframe_camelyon_challenge_image.dcm')
    )
    path = _dicom_series_path(dcm)
    store_path = str(path.GetStorePath())
    with dicom_store_mock.MockDicomStores(store_path) as mk_dicom_stores:
      mk_dicom_stores[store_path].add_instance(dcm)

      request = {
          _EndpointJsonKeys.INSTANCES: [
              # Good Request
              _dicom_instance_patch_request(
                  path, dcm.SOPInstanceUID, 1, 2, 3, 4
              ),
              # Bad Request
              _dicom_instance_patch_request(path, '1.2.3', 5, 6, 7, 8),
              # Good Request
              _dicom_instance_patch_request(
                  path, dcm.SOPInstanceUID, 9, 10, 11, 12
              ),
          ]
      }
      with pete_predictor_v2.PetePredictor() as predictor:
        results = predictor.predict(request, _mock_model_runner)
      self.assertEqual(
          results,
          {
              'predictions': [
                  {
                      'result': {
                          'patch_embeddings': [
                              {
                                  'embedding_vector': [
                                      0.7732688784599304,
                                      0.7125012278556824,
                                      0.8252677321434021,
                                  ],
                                  'patch_coordinate': {
                                      'x_origin': 1,
                                      'y_origin': 2,
                                      'width': 224,
                                      'height': 224,
                                  },
                              },
                              {
                                  'embedding_vector': [
                                      0.771074116230011,
                                      0.7090619206428528,
                                      0.8237756490707397,
                                  ],
                                  'patch_coordinate': {
                                      'x_origin': 3,
                                      'y_origin': 4,
                                      'width': 224,
                                      'height': 224,
                                  },
                              },
                          ]
                      },
                  },
                  {
                      'error': {
                          'code': 'DICOM_PATH_ERROR',
                          'description': (
                              'Could not find DICOM imaging; path:'
                              ' https://healthcare.googleapis.com/v1/projects/project/locations/location/datasets/dataset/dicomStores/dicomstore/dicomWeb/studies/1.3.6.1.4.1.11129.5.7.999.18649109954048068.740.1688792381777315/series/1.3.6.1.4.1.11129.5.7.0.1.517182092386.24422120.1688792467737634.'
                          ),
                      },
                  },
                  {
                      'result': {
                          'patch_embeddings': [
                              {
                                  'embedding_vector': [
                                      0.7646647095680237,
                                      0.6990870237350464,
                                      0.8194893002510071,
                                  ],
                                  'patch_coordinate': {
                                      'x_origin': 9,
                                      'y_origin': 10,
                                      'width': 224,
                                      'height': 224,
                                  },
                              },
                              {
                                  'embedding_vector': [
                                      0.7625885009765625,
                                      0.6958702802658081,
                                      0.818108856678009,
                                  ],
                                  'patch_coordinate': {
                                      'x_origin': 11,
                                      'y_origin': 12,
                                      'width': 224,
                                      'height': 224,
                                  },
                              },
                          ]
                      },
                  },
              ]
          },
      )

  @parameterized.named_parameters([
      dict(testcase_name='monochrome_1', shape=(224, 224)),
      dict(testcase_name='monochrome_2', shape=(224, 224, 1)),
      dict(testcase_name='rgb', shape=(224, 224, 3)),
      dict(testcase_name='rgba', shape=(224, 224, 4)),
  ])
  def test_fetch_patch_bytes_norms_monochrome_images_to_three_channels(
      self, shape
  ):
    mem = np.zeros(shape=shape, dtype=np.uint8)
    mock_patch = mock.create_autospec(dicom_slide.DicomPatch, instance=True)
    mock_patch.image_bytes.return_value = mem
    self.assertEqual(
        pete_predictor_v2._fetch_patch_bytes(mock_patch, None).shape,
        (1, 224, 224, 3),
    )

  @flagsaver.flagsaver(approved_gcs_source_list=['gs://abc', 'gs://123'])
  def test_validate_gcs_image_source_raises(self):
    with self.assertRaises(pete_errors.UnapprovedGcsBucketError):
      pete_predictor_v2._validate_gcs_image_source(
          'gs://test_bucket/google.png'
      )

  @parameterized.parameters(['gs://abc/google.png', 'gs://123/google.png'])
  @flagsaver.flagsaver(approved_gcs_source_list=['gs://abc', 'gs://123'])
  def test_validate_gcs_image_source_valid(self, source):
    self.assertIsNone(pete_predictor_v2._validate_gcs_image_source(source))

  def test_validate_default_gcs_image_source_valid(self):
    self.assertIsNone(
        pete_predictor_v2._validate_gcs_image_source(
            'gs://test_bucket/google.png'
        )
    )

  @flagsaver.flagsaver(
      approved_dicom_store_source_list=['http://abc', 'http://123']
  )
  def test_validate_dicom_image_source_raises(self):
    with self.assertRaises(pete_errors.UnapprovedDicomStoreError):
      pete_predictor_v2._validate_dicom_image_source(
          'http://test_bucket/google.png'
      )

  @parameterized.parameters(['http://abc/studies', 'http://123/studies'])
  @flagsaver.flagsaver(
      approved_dicom_store_source_list=['http://abc', 'http://123']
  )
  def test_validate_dicom_image_source_valid(self, source):
    self.assertIsNone(pete_predictor_v2._validate_dicom_image_source(source))

  def test_validate_default_dicom_image_source_valid(self):
    self.assertIsNone(
        pete_predictor_v2._validate_dicom_image_source(
            'http://test_bucket/studies'
        )
    )


if __name__ == '__main__':
  absltest.main()
