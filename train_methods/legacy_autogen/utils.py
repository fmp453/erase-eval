import functools
import inspect
import json
from typing import Callable, Literal, TypedDict, Any, Annotated, ForwardRef
from typing_extensions import get_args, get_origin

from pydantic import BaseModel
from pydantic._internal._typing_extra import try_eval_type


class UserMessageTextContentPart(TypedDict):
    type: Literal["text"]
    text: str

class UserMessageImageContentPart(TypedDict):
    type: Literal["image_url"]
    image_url: dict[Literal["url"], str]


def content_str(content: str | list[UserMessageTextContentPart | UserMessageImageContentPart] | None) -> str:
    """Converts the `content` field of an OpenAI message into a string format.

    This function processes content that may be a string, a list of mixed text and image URLs, or None,
    and converts it into a string. Text is directly appended to the result string, while image URLs are
    represented by a placeholder image token. If the content is None, an empty string is returned.

    Args:
        - content (Union[str, List, None]): The content to be processed. Can be a string, a list of dictionaries
                                      representing text and image URLs, or None.

    Returns:
        str: A string representation of the input content. Image URLs are replaced with an image token.

    Note:
    - The function expects each dictionary in the list to have a "type" key that is either "text" or "image_url".
      For "text" type, the "text" key's value is appended to the result. For "image_url", an image token is appended.
    - This function is useful for handling content that may include both text and image references, especially
      in contexts where images need to be represented as placeholders.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError(f"content must be None, str, or list, but got {type(content)}")

    rst = ""
    for item in content:
        if not isinstance(item, dict):
            raise TypeError("Wrong content format: every element should be dict if the content is a list.")
        assert "type" in item, "Wrong content format. Missing 'type' key in content's dict."
        if item["type"] == "text":
            rst += item["text"]
        elif item["type"] == "image_url":
            rst += "<image>"
        else:
            raise ValueError(f"Wrong content format: unknown type {item['type']} within the content")
    return rst

def get_typed_annotation(annotation: Any, globalns: dict[str, Any]) -> Any:
    """Get the type annotation of a parameter.

    Args:
        annotation: The annotation of the parameter
        globalns: The global namespace of the function

    Returns:
        The type annotation of the parameter
    """
    if isinstance(annotation, str):
        annotation = ForwardRef(annotation)
        annotation = try_eval_type(annotation, globalns, globalns)
    return annotation

def get_typed_signature(call: Callable[..., Any]) -> inspect.Signature:
    """Get the signature of a function with type annotations.

    Args:
        call: The function to get the signature for

    Returns:
        The signature of the function with type annotations
    """
    signature = inspect.signature(call)
    globalns = getattr(call, "__globals__", {})
    typed_params = [
        inspect.Parameter(
            name=param.name,
            kind=param.kind,
            default=param.default,
            annotation=get_typed_annotation(param.annotation, globalns),
        )
        for param in signature.parameters.values()
    ]
    typed_signature = inspect.Signature(typed_params)
    return typed_signature

def get_param_annotations(typed_signature: inspect.Signature) -> dict[str, Annotated[type[Any], str] | type[Any]]:
    """Get the type annotations of the parameters of a function

    Args:
        typed_signature: The signature of the function with type annotations

    Returns:
        A dictionary of the type annotations of the parameters of the function
    """
    return {
        k: v.annotation for k, v in typed_signature.parameters.items() if v.annotation is not inspect.Signature.empty
    }

def get_load_param_if_needed_function(t: Any) -> Callable[[dict[str, Any], type[BaseModel]], BaseModel] | None:
    """Get a function to load a parameter if it is a Pydantic model

    Args:
        t: The type annotation of the parameter

    Returns:
        A function to load the parameter if it is a Pydantic model, otherwise None

    """
    if get_origin(t) is Annotated:
        return get_load_param_if_needed_function(get_args(t)[0])

    def load_base_model(v: dict[str, Any], t: type[BaseModel]) -> BaseModel:
        return t(**v)

    return load_base_model if isinstance(t, type) and issubclass(t, BaseModel) else None


def load_basemodels_if_needed(func: Callable[..., Any]) -> Callable[..., Any]:
    """A decorator to load the parameters of a function if they are Pydantic models

    Args:
        func: The function with annotated parameters

    Returns:
        A function that loads the parameters before calling the original function

    """
    # get the type annotations of the parameters
    typed_signature = get_typed_signature(func)
    param_annotations = get_param_annotations(typed_signature)

    # get functions for loading BaseModels when needed based on the type annotations
    kwargs_mapping_with_nones = {k: get_load_param_if_needed_function(t) for k, t in param_annotations.items()}

    # remove the None values
    kwargs_mapping = {k: f for k, f in kwargs_mapping_with_nones.items() if f is not None}

    # a function that loads the parameters before calling the original function
    @functools.wraps(func)
    def _load_parameters_if_needed(*args: Any, **kwargs: Any) -> Any:
        # load the BaseModels if needed
        for k, f in kwargs_mapping.items():
            kwargs[k] = f(kwargs[k], param_annotations[k])

        # call the original function
        return func(*args, **kwargs)

    @functools.wraps(func)
    async def _a_load_parameters_if_needed(*args: Any, **kwargs: Any) -> Any:
        # load the BaseModels if needed
        for k, f in kwargs_mapping.items():
            kwargs[k] = f(kwargs[k], param_annotations[k])

        # call the original function
        return await func(*args, **kwargs)

    if inspect.iscoroutinefunction(func):
        return _a_load_parameters_if_needed
    else:
        return _load_parameters_if_needed


def serialize_to_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    elif isinstance(x, BaseModel):
        return x.model_dump_json()
    else:
        return json.dumps(x, ensure_ascii=False)

