from typing import Optional, Dict

from pydantic import BaseModel
from pydantic.fields import ModelField


def params_validate_types_match(
    params: BaseModel, values: Dict, field: Optional[ModelField] = None
):
    """
    Custom pydantic validator used together with :func:`pydantic.validator`.
    Can be used for :class:`BaseModel`'s that contain both a type and params attribute.
    It checks if the parameter set contains a type attribute and if it matches the type
    specified in the model itself.

    :param params: Model of parameter set
    :param values: Dictionary with previously checked attributes. Has to contain key "type"
    :params field: Supplied by pydantic, set to None for easier testability
    :return: params if they are valid
    :raises ValueError, TypeError
    """
    # Do not throw error but accept empty params. This allows one to
    # not specify params if the type allows it and e.g. just uses default values
    if field and not field.required and params is None:
        return params

    try:
        expected_type = values["type"]
        params_type = getattr(params, "type")
    except KeyError:
        raise ValueError(f'Required field "type" not given.')
    except AttributeError:
        raise ValueError(
            f'Field "type" is missing in the class definition of model {params.__class__}'
        )

    if expected_type != params_type:
        raise TypeError(
            f"Given values for parameters of type {params_type} do not match the expected type {expected_type}"
        )
    return params
