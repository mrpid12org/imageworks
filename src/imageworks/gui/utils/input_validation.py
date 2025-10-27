"""Input validation helpers for Streamlit forms."""

import streamlit as st
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

from imageworks.gui.utils.error_handling import (
    validate_path,
    validate_url,
    validate_threshold,
    ValidationError,
    handle_error,
)


def validated_text_input(
    label: str,
    value: str = "",
    help: Optional[str] = None,
    key: Optional[str] = None,
    required: bool = False,
    **kwargs,
) -> Optional[str]:
    """
    Text input with validation display.

    Args:
        label: Input label
        value: Default value
        help: Help text
        key: Session state key
        required: If True, value cannot be empty
        **kwargs: Additional st.text_input arguments

    Returns:
        Validated input value or None if invalid
    """
    input_value = st.text_input(label, value=value, help=help, key=key, **kwargs)

    # Validation
    if required and not input_value:
        st.error(f"⚠️ {label} is required")
        return None

    return input_value


def validated_path_input(
    label: str,
    value: str = "",
    help: Optional[str] = None,
    key: Optional[str] = None,
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    extensions: Optional[List[str]] = None,
    **kwargs,
) -> Optional[Path]:
    """
    Path input with validation display.

    Args:
        label: Input label
        value: Default value
        help: Help text
        key: Session state key
        must_exist: If True, path must exist
        must_be_file: If True, must be file
        must_be_dir: If True, must be directory
        extensions: Valid file extensions
        **kwargs: Additional st.text_input arguments

    Returns:
        Validated Path object or None if invalid
    """
    input_value = st.text_input(label, value=value, help=help, key=key, **kwargs)

    if not input_value:
        return None

    try:
        path = validate_path(
            input_value,
            must_exist=must_exist,
            must_be_file=must_be_file,
            must_be_dir=must_be_dir,
            extensions=extensions,
        )

        # Show success indicator for existing paths
        if must_exist and path.exists():
            st.success("✅ Path exists")

        return path

    except ValidationError as e:
        st.error(f"⚠️ {str(e)}")
        return None


def validated_url_input(
    label: str,
    value: str = "",
    help: Optional[str] = None,
    key: Optional[str] = None,
    **kwargs,
) -> Optional[str]:
    """
    URL input with validation display.

    Args:
        label: Input label
        value: Default value
        help: Help text
        key: Session state key
        **kwargs: Additional st.text_input arguments

    Returns:
        Validated URL or None if invalid
    """
    input_value = st.text_input(label, value=value, help=help, key=key, **kwargs)

    if not input_value:
        return None

    try:
        url = validate_url(input_value)
        st.success("✅ Valid URL")
        return url

    except ValidationError as e:
        st.error(f"⚠️ {str(e)}")
        return None


def validated_number_input(
    label: str,
    value: float,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    help: Optional[str] = None,
    key: Optional[str] = None,
    validate_range: bool = True,
    **kwargs,
) -> Optional[float]:
    """
    Number input with validation display.

    Args:
        label: Input label
        value: Default value
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        help: Help text
        key: Session state key
        validate_range: If True, validate against min/max
        **kwargs: Additional st.number_input arguments

    Returns:
        Validated number or None if invalid
    """
    input_value = st.number_input(
        label,
        value=value,
        min_value=min_value,
        max_value=max_value,
        help=help,
        key=key,
        **kwargs,
    )

    if validate_range and min_value is not None and max_value is not None:
        try:
            validate_threshold(input_value, min_value, max_value)
            return input_value
        except ValidationError as e:
            st.error(f"⚠️ {str(e)}")
            return None

    return input_value


def validated_selectbox(
    label: str,
    options: List[Any],
    index: int = 0,
    help: Optional[str] = None,
    key: Optional[str] = None,
    allow_none: bool = False,
    **kwargs,
) -> Optional[Any]:
    """
    Selectbox with validation display.

    Args:
        label: Input label
        options: List of options
        index: Default index
        help: Help text
        key: Session state key
        allow_none: If True, allow None selection
        **kwargs: Additional st.selectbox arguments

    Returns:
        Selected option or None if invalid
    """
    if not options:
        st.error(f"⚠️ No options available for {label}")
        return None

    selected = st.selectbox(
        label, options=options, index=index, help=help, key=key, **kwargs
    )

    if not allow_none and selected is None:
        st.error(f"⚠️ {label} is required")
        return None

    return selected


def show_validation_summary(errors: List[str], warnings: List[str] = None) -> bool:
    """
    Display validation summary.

    Args:
        errors: List of error messages
        warnings: List of warning messages

    Returns:
        True if no errors, False otherwise
    """
    has_errors = len(errors) > 0
    has_warnings = warnings and len(warnings) > 0

    if has_errors or has_warnings:
        with st.expander("⚠️ Validation Issues", expanded=has_errors):
            if has_errors:
                st.markdown("### ❌ Errors")
                for error in errors:
                    st.error(error)

            if has_warnings:
                st.markdown("### ⚠️ Warnings")
                for warning in warnings:
                    st.warning(warning)

    return not has_errors


def validate_form_data(
    data: Dict[str, Any],
    required_fields: List[str],
    field_validators: Dict[str, Callable] = None,
) -> tuple[bool, List[str]]:
    """
    Validate form data against requirements.

    Args:
        data: Form data dictionary
        required_fields: List of required field names
        field_validators: Optional dict mapping field names to validator functions

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check required fields
    for field in required_fields:
        if field not in data or not data[field]:
            errors.append(f"Required field missing: {field}")

    # Run custom validators
    if field_validators:
        for field, validator in field_validators.items():
            if field in data:
                try:
                    validator(data[field])
                except ValidationError as e:
                    errors.append(f"{field}: {str(e)}")

    return len(errors) == 0, errors


def render_validation_badge(is_valid: bool, message: str = "") -> None:
    """
    Render a validation status badge.

    Args:
        is_valid: Whether validation passed
        message: Optional message to display
    """
    if is_valid:
        st.success(f"✅ {message or 'Valid'}")
    else:
        st.error(f"❌ {message or 'Invalid'}")


def safe_execute_with_validation(
    func: Callable,
    validation_checks: List[Callable],
    error_message: str = "Validation failed",
) -> Optional[Any]:
    """
    Execute function only if all validation checks pass.

    Args:
        func: Function to execute
        validation_checks: List of validation functions that raise ValidationError
        error_message: Error message prefix

    Returns:
        Function result or None if validation failed
    """
    try:
        # Run all validation checks
        for check in validation_checks:
            check()

        # Execute function
        return func()

    except ValidationError as e:
        handle_error(e, error_message)
        return None
    except Exception as e:
        handle_error(e, f"{error_message} - Unexpected error")
        return None
