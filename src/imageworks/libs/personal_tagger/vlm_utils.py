"""VLM (Vision-Language Model) utilities and operations.

Provides common utilities for VLM inference, prompt management, response parsing,
and model orchestration across personal tagger applications.
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class VLMPromptTemplate:
    """Template for VLM prompts with parameter substitution."""

    name: str
    template: str
    required_params: List[str] = field(default_factory=list)
    optional_params: List[str] = field(default_factory=list)
    description: str = ""

    def format(self, **kwargs) -> str:
        """Format template with provided parameters.

        Args:
            **kwargs: Parameters for template substitution

        Returns:
            Formatted prompt string
        """
        # Check required parameters
        missing = [p for p in self.required_params if p not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        # Add defaults for optional parameters
        for param in self.optional_params:
            if param not in kwargs:
                kwargs[param] = "Not available"

        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Template parameter not provided: {e}")


@dataclass
class VLMModelConfig:
    """Configuration for VLM model inference."""

    model_name: str
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    max_tokens: int = 300
    temperature: float = 0.1
    timeout: int = 120
    retry_attempts: int = 3
    retry_delay: float = 1.0


class VLMPromptManager:
    """Manager for VLM prompt templates and operations."""

    def __init__(self):
        self.templates: Dict[str, VLMPromptTemplate] = {}
        self._register_default_templates()

    def register_template(self, template: VLMPromptTemplate) -> None:
        """Register a new prompt template.

        Args:
            template: Template to register
        """
        self.templates[template.name] = template
        logger.debug(f"Registered template: {template.name}")

    def get_template(self, name: str) -> Optional[VLMPromptTemplate]:
        """Get a registered template by name.

        Args:
            name: Template name

        Returns:
            Template if found, None otherwise
        """
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List all registered template names.

        Returns:
            List of template names
        """
        return list(self.templates.keys())

    def _register_default_templates(self) -> None:
        """Register default prompt templates."""

        # Color narration template
        color_narration = VLMPromptTemplate(
            name="color_narration",
            template="""You are analyzing a photograph that should be monochrome (black and white) but contains some residual color.

The image has been analyzed and found to have:
- Hue distribution: {hue_analysis}
- Chroma levels: {chroma_analysis}
- Color contamination level: {contamination_level}

Please describe in natural language where you observe residual color in this image. Focus on:
1. Specific regions or objects that show color
2. The type of color cast (warm/cool, specific hues)
3. Whether the color appears intentional or accidental

Provide a concise, professional description suitable for metadata (max 200 words).""",
            required_params=["hue_analysis", "chroma_analysis", "contamination_level"],
            description="Template for generating color narration descriptions",
        )
        self.register_template(color_narration)

        # Color validation template
        color_validation = VLMPromptTemplate(
            name="color_validation",
            template="""You are validating an existing color description against visual evidence in an image.

Existing description: "{existing_description}"
Analysis data: {analysis_data}

Please evaluate:
1. Does the description accurately match what you observe?
2. Are there color areas mentioned that you don't see?
3. Are there visible color areas not mentioned in the description?

Respond with "VALID" if the description is accurate, or "INVALID" followed by corrections.""",
            required_params=["existing_description", "analysis_data"],
            description="Template for validating existing color descriptions",
        )
        self.register_template(color_validation)

        # Generic image analysis template
        image_analysis = VLMPromptTemplate(
            name="image_analysis",
            template="""Analyze this image focusing on: {focus_areas}

{additional_context}

Provide a detailed analysis addressing the specified focus areas.""",
            required_params=["focus_areas"],
            optional_params=["additional_context"],
            description="Generic template for image analysis tasks",
        )
        self.register_template(image_analysis)


class VLMResponseParser:
    """Parser for structured VLM response extraction."""

    @staticmethod
    def extract_color_regions(text: str) -> List[str]:
        """Extract color region mentions from response text.

        Args:
            text: VLM response text

        Returns:
            List of identified color regions
        """
        text_lower = text.lower()

        # Common region keywords to look for
        region_keywords = [
            "background",
            "foreground",
            "subject",
            "person",
            "face",
            "skin",
            "hair",
            "clothing",
            "shirt",
            "dress",
            "pants",
            "jacket",
            "sky",
            "clouds",
            "horizon",
            "ground",
            "floor",
            "wall",
            "water",
            "trees",
            "vegetation",
            "flowers",
            "leaves",
            "shadows",
            "highlights",
            "midtones",
            "edges",
            "corners",
            "left side",
            "right side",
            "center",
            "top",
            "bottom",
            "fabric",
            "metal",
            "glass",
            "wood",
            "stone",
        ]

        found_regions = []
        for keyword in region_keywords:
            if keyword in text_lower:
                found_regions.append(keyword)

        return found_regions

    @staticmethod
    def extract_confidence_indicators(text: str) -> float:
        """Estimate confidence from response text indicators.

        Args:
            text: VLM response text

        Returns:
            Estimated confidence score (0.0-1.0)
        """
        text_lower = text.lower()

        # High confidence indicators
        high_conf_words = [
            "clear",
            "obvious",
            "distinct",
            "strong",
            "prominent",
            "definite",
        ]
        # Medium confidence indicators
        med_conf_words = ["subtle", "slight", "mild", "moderate", "some"]
        # Low confidence indicators
        low_conf_words = ["minimal", "trace", "barely", "faint", "very slight"]
        # Uncertainty indicators
        uncertain_words = ["might", "could", "appears", "seems", "possibly", "perhaps"]

        scores = []

        for word in high_conf_words:
            if word in text_lower:
                scores.append(0.9)

        for word in med_conf_words:
            if word in text_lower:
                scores.append(0.7)

        for word in low_conf_words:
            if word in text_lower:
                scores.append(0.5)

        for word in uncertain_words:
            if word in text_lower:
                scores.append(0.6)

        # Return average if we found indicators, otherwise default
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0.8  # Default confidence

    @staticmethod
    def extract_validation_result(text: str) -> Dict[str, Any]:
        """Extract validation result from response text.

        Args:
            text: VLM validation response

        Returns:
            Dictionary with validation results
        """
        text_upper = text.upper()

        result = {
            "is_valid": "VALID" in text_upper,
            "corrections": "",
            "confidence": 0.8,
        }

        if not result["is_valid"] and "INVALID" in text_upper:
            # Extract corrections after INVALID
            invalid_pos = text_upper.find("INVALID")
            if invalid_pos != -1:
                corrections = text[invalid_pos + 7 :].strip()
                result["corrections"] = corrections

        # Estimate confidence based on certainty of language
        if "definitely" in text.lower() or "clearly" in text.lower():
            result["confidence"] = 0.95
        elif "possibly" in text.lower() or "might" in text.lower():
            result["confidence"] = 0.6

        return result


class VLMBatchProcessor:
    """Batch processing utilities for VLM inference."""

    def __init__(
        self,
        config: VLMModelConfig,
        max_concurrent: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """Initialize batch processor.

        Args:
            config: VLM model configuration
            max_concurrent: Maximum concurrent requests
            progress_callback: Optional progress reporting callback
        """
        self.config = config
        self.max_concurrent = max_concurrent
        self.progress_callback = progress_callback

    async def process_batch_async(
        self, requests: List[Dict[str, Any]], session: aiohttp.ClientSession
    ) -> List[Dict[str, Any]]:
        """Process a batch of requests asynchronously.

        Args:
            requests: List of VLM request dictionaries
            session: HTTP session for requests

        Returns:
            List of response dictionaries
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_single(request_data: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await self._make_async_request(request_data, session)

        # Create tasks for all requests
        tasks = [process_single(req) for req in requests]

        # Process with progress tracking
        results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            results.append(result)

            if self.progress_callback:
                self.progress_callback(i + 1, len(tasks))

        return results

    async def _make_async_request(
        self, request_data: Dict[str, Any], session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        """Make single async VLM request.

        Args:
            request_data: Request data dictionary
            session: HTTP session

        Returns:
            Response dictionary
        """
        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model_name,
            "messages": request_data["messages"],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": False,
        }

        for attempt in range(self.config.retry_attempts):
            try:
                timeout = aiohttp.ClientTimeout(total=self.config.timeout)
                async with session.post(
                    url, json=payload, headers=headers, timeout=timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "success": True,
                            "response": result,
                            "request_data": request_data,
                        }
                    else:
                        error_text = await response.text()
                        logger.warning(
                            f"VLM request failed with status {response.status}: {error_text}"
                        )

            except Exception as e:
                logger.warning(f"VLM request attempt {attempt + 1} failed: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (2**attempt))

        return {
            "success": False,
            "error": "Max retry attempts exceeded",
            "request_data": request_data,
        }

    def process_batch_sync(
        self, requests: List[Dict[str, Any]], use_threading: bool = True
    ) -> List[Dict[str, Any]]:
        """Process batch synchronously with optional threading.

        Args:
            requests: List of VLM request dictionaries
            use_threading: Whether to use thread pool for concurrency

        Returns:
            List of response dictionaries
        """
        if use_threading:
            return self._process_with_threading(requests)
        else:
            return self._process_sequential(requests)

    def _process_with_threading(
        self, requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process requests using thread pool."""
        results = [None] * len(requests)

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all requests
            future_to_index = {
                executor.submit(self._make_sync_request, req): i
                for i, req in enumerate(requests)
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_index)):
                index = future_to_index[future]
                results[index] = future.result()

                if self.progress_callback:
                    self.progress_callback(i + 1, len(requests))

        return results

    def _process_sequential(
        self, requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process requests sequentially."""
        results = []

        for i, request in enumerate(requests):
            result = self._make_sync_request(request)
            results.append(result)

            if self.progress_callback:
                self.progress_callback(i + 1, len(requests))

        return results

    def _make_sync_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make single synchronous VLM request."""
        # This would use requests library for sync HTTP
        # Implementation would mirror async version but with requests
        # For now, return placeholder
        return {
            "success": False,
            "error": "Sync implementation not yet available",
            "request_data": request_data,
        }


class VLMModelManager:
    """Manager for VLM model lifecycle and configuration."""

    def __init__(self):
        self.configs: Dict[str, VLMModelConfig] = {}
        self.active_config: Optional[str] = None

    def register_model(self, name: str, config: VLMModelConfig) -> None:
        """Register a VLM model configuration.

        Args:
            name: Model configuration name
            config: Model configuration
        """
        self.configs[name] = config

        if self.active_config is None:
            self.active_config = name

    def set_active_model(self, name: str) -> None:
        """Set the active model configuration.

        Args:
            name: Model configuration name
        """
        if name not in self.configs:
            raise ValueError(f"Model configuration '{name}' not found")

        self.active_config = name

    def get_active_config(self) -> Optional[VLMModelConfig]:
        """Get the active model configuration.

        Returns:
            Active model configuration or None
        """
        if self.active_config:
            return self.configs[self.active_config]
        return None

    def list_models(self) -> List[str]:
        """List all registered model names.

        Returns:
            List of model configuration names
        """
        return list(self.configs.keys())

    async def health_check(self, model_name: Optional[str] = None) -> bool:
        """Check if specified model is available and responding.

        Args:
            model_name: Model to check (default: active model)

        Returns:
            True if model is healthy
        """
        config_name = model_name or self.active_config
        if not config_name or config_name not in self.configs:
            return False

        config = self.configs[config_name]

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = f"{config.base_url}/models"
                headers = {"Authorization": f"Bearer {config.api_key}"}

                async with session.get(url, headers=headers) as response:
                    return response.status == 200
        except Exception:
            return False
