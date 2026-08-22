"""The glossary ships dark. These tests hold that default in place."""

import pytest

from config import Settings


def test_glossary_is_disabled_by_default():
    settings = Settings(_env_file=None)
    assert settings.glossary_enabled is False


def test_enabling_without_an_admin_key_fails_at_startup():
    # The admin router is the only authorization boundary in this deployment,
    # so a missing key must fail loudly at boot rather than at the first write.
    with pytest.raises(ValueError, match="TG_ADMIN_API_KEY"):
        Settings(_env_file=None, glossary_enabled=True, admin_api_key=None)


def test_enabling_with_an_admin_key_is_accepted():
    settings = Settings(_env_file=None, glossary_enabled=True, admin_api_key="secret")
    assert settings.glossary_enabled is True
    assert settings.admin_api_key == "secret"


def test_unknown_domain_policy_defaults_to_reject():
    settings = Settings(_env_file=None)
    assert settings.glossary_unknown_domain == "reject"


def test_an_invalid_unknown_domain_policy_fails_at_startup():
    with pytest.raises(ValueError):
        Settings(_env_file=None, glossary_unknown_domain="ignore")


def test_terminology_mode_defaults_to_enforce():
    settings = Settings(_env_file=None)
    assert settings.terminology_mode == "enforce"


def test_an_invalid_terminology_mode_fails_at_startup():
    # TG_TERMINOLOGY_MODE=enforcee used to boot happily and silently behave
    # like every value that is not the exact string "off" -- indistinguishable
    # from a correctly configured "enforce".
    with pytest.raises(ValueError):
        Settings(_env_file=None, terminology_mode="enforcee")
