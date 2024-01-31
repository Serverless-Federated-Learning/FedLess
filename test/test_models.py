import pytest
from pydantic import ValidationError, BaseModel, validator

from fedless.common.models import params_validate_types_match


class Model(BaseModel):
    """Example Parameters"""

    type: str = "aws"
    attribute: bool


class MetaModel(BaseModel):
    """Meta config that wraps parameters"""

    type: str
    params: Model

    _params_type_matches_type = validator("params", allow_reuse=True)(
        params_validate_types_match
    )


def test_params_match_type():
    model = Model(attribute=False)
    values = {"type": "aws"}
    returned_model = params_validate_types_match(model, values)
    assert returned_model == model


def test_params_match_type_fails_without_type():
    model = Model(attribute=False)
    values = {}
    with pytest.raises(ValueError):
        params_validate_types_match(model, values)


def test_params_match_type_fails_without_type_in_model():
    class ModelWithoutType(BaseModel):
        attribute: bool

    model = ModelWithoutType(attribute=False)
    values = {"type": "aws"}
    with pytest.raises(ValueError):
        params_validate_types_match(model, values)


def test_params_match_type_fails_for_different_types():
    model = Model(attribute=False)
    values = {"type": "local"}
    with pytest.raises(TypeError):
        params_validate_types_match(model, values)


def test_pydantic_always_wraps_error_in_validator():
    with pytest.raises(ValidationError):
        MetaModel(type="not-aws", params=Model(attribute=False))

    with pytest.raises(ValidationError):
        MetaModel.parse_obj({"params": {"attribute": True}})

    with pytest.raises(ValidationError):
        MetaModel.parse_obj({"type": "not-aws", "params": {"attribute": True}})

    with pytest.raises(ValidationError):
        MetaModel.parse_obj(
            {"type": "aws", "params": {"type": "not-aws", "attribute": True}}
        )


def test_validator_accepts_objects():
    model = MetaModel(type="aws", params=Model(type="aws", attribute=False))
    model_from_dict = MetaModel.parse_obj(
        {"type": "aws", "params": {"attribute": False}}
    )
    assert model.json() == model_from_dict.json()
