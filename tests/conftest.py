import pytest

from askmydocs.config import load_settings


@pytest.fixture(scope="session")
def offline_settings():
    return load_settings(profile="offline")


@pytest.fixture(scope="session")
def rag(offline_settings):
    from askmydocs.pipeline import RAGPipeline

    return RAGPipeline(offline_settings, log=lambda *a: None)
