"""
Extension metadata configuration.
"""
from typing import Dict, Any


class ExtensionMetadata:
    """Extension metadata class with validation and type hints."""
    
    PACKAGE_NAME = "extension_kokoro_tts_api"
    NAME = "Kokoro TTS API"
    VERSION = "1.0.0"
    DESCRIPTION = "OpenAI-compatible Text-to-Speech API using Kokoro TTS and other models"
    AUTHOR = "hexgrad"
    EXTENSION_AUTHOR = "rsxdalv"
    LICENSE = "MIT"
    
    # URLs
    WEBSITE = "https://huggingface.co/hexgrad/Kokoro-82M"
    EXTENSION_WEBSITE = "https://github.com/rsxdalv/extension_kokoro_tts_api"
    REQUIREMENTS_URL = "git+https://github.com/rsxdalv/extension_kokoro_tts_api@main"
    
    # Extension classification
    EXTENSION_TYPE = "interface"
    EXTENSION_CLASS = "tools"
    PLATFORM_VERSION = "0.0.1"
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert metadata to dictionary format expected by TTS WebUI."""
        return {
            "package_name": cls.PACKAGE_NAME,
            "name": cls.NAME,
            "version": cls.VERSION,
            "requirements": cls.REQUIREMENTS_URL,
            "description": cls.DESCRIPTION,
            "extension_type": cls.EXTENSION_TYPE,
            "extension_class": cls.EXTENSION_CLASS,
            "author": cls.AUTHOR,
            "extension_author": cls.EXTENSION_AUTHOR,
            "license": cls.LICENSE,
            "website": cls.WEBSITE,
            "extension_website": cls.EXTENSION_WEBSITE,
            "extension_platform_version": cls.PLATFORM_VERSION,
        }
    
    @classmethod
    def get_display_info(cls) -> Dict[str, str]:
        """Get user-friendly display information."""
        return {
            "name": cls.NAME,
            "version": cls.VERSION,
            "description": cls.DESCRIPTION,
            "author": f"{cls.AUTHOR} (extension by {cls.EXTENSION_AUTHOR})",
            "website": cls.WEBSITE,
        }


def get_extension_metadata() -> Dict[str, Any]:
    """
    Get extension metadata in the format expected by TTS WebUI.
    
    This function replaces the old extension_tts_generation_webui_metadata().
    """
    return ExtensionMetadata.to_dict()


# Legacy compatibility
def extension_tts_generation_webui_metadata() -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    return get_extension_metadata()
