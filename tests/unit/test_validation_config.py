"""
Comprehensive unit tests for validation models and configuration settings.

Tests:
- QueryInput validation and sanitization
- MCTSConfig parameter bounds
- XSS prevention (script tags blocked)
- Template injection prevention
- Whitespace normalization
- Settings initialization and environment variables
- SecretStr handling
- Provider enum validation
"""

import os
import sys
import warnings
from unittest.mock import patch

import pytest

pytest.importorskip("pydantic", reason="pydantic required for settings/benchmark")

sys.path.insert(0, ".")

from pydantic import SecretStr, ValidationError

from src.config.settings import (
    LLMProvider,
    LogLevel,
    Settings,
    get_settings,
    reset_settings,
)
from src.models.validation import (
    MAX_BATCH_SIZE,
    MAX_ITERATIONS,
    MAX_QUERY_LENGTH,
    MIN_QUERY_LENGTH,
    AgentConfig,
    APIRequestMetadata,
    BatchQueryInput,
    FileReadInput,
    MCPToolInput,
    MCTSConfig,
    QueryInput,
    RAGConfig,
    WebFetchInput,
    validate_mcts_config,
    validate_query,
    validate_tool_input,
)


class TestQueryInputValidation:
    """Test suite for QueryInput validation and sanitization."""

    def test_valid_simple_query(self):
        """Test valid simple query passes validation."""
        query_input = QueryInput(query="What is the weather today?")
        assert query_input.query == "What is the weather today?"

    def test_valid_query_with_options(self):
        """Test valid query with all options."""
        query_input = QueryInput(
            query="Complex question here", use_rag=False, use_mcts=True, thread_id="test-thread-123"
        )
        assert query_input.query == "Complex question here"
        assert query_input.use_rag is False
        assert query_input.use_mcts is True
        assert query_input.thread_id == "test-thread-123"

    def test_default_values(self):
        """Test default values are applied correctly."""
        query_input = QueryInput(query="Test query")
        assert query_input.use_rag is True
        assert query_input.use_mcts is False
        assert query_input.thread_id is None

    @pytest.mark.parametrize(
        "whitespace_query,expected",
        [
            ("  test query  ", "test query"),
            ("\nquery with newlines\n", "query with newlines"),
            ("\tquery with tabs\t", "query with tabs"),
            ("multiple   spaces   here", "multiple spaces here"),
            ("  multiple   whitespace   everywhere  ", "multiple whitespace everywhere"),
        ],
    )
    def test_whitespace_normalization(self, whitespace_query, expected):
        """Test that whitespace is properly normalized."""
        query_input = QueryInput(query=whitespace_query)
        assert query_input.query == expected

    def test_null_byte_removal(self):
        """Test that null bytes are removed from query."""
        query_input = QueryInput(query="test\x00query\x00here")
        assert "\x00" not in query_input.query
        # Null bytes are removed, then consecutive whitespace is normalized
        assert query_input.query == "testqueryhere"

    @pytest.mark.parametrize(
        "empty_query",
        [
            "",
            "   ",
            "\n\n\n",
            "\t\t",
            "  \n  \t  ",
        ],
    )
    def test_empty_query_rejected(self, empty_query):
        """Test that empty or whitespace-only queries are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            QueryInput(query=empty_query)
        error_msg = str(exc_info.value).lower()
        # Check for various validation error messages
        assert "empty" in error_msg or "whitespace" in error_msg or "at least 1 character" in error_msg

    def test_query_minimum_length(self):
        """Test query must meet minimum length."""
        # Single character should work
        query_input = QueryInput(query="a")
        assert len(query_input.query) >= MIN_QUERY_LENGTH

    def test_query_maximum_length(self):
        """Test query cannot exceed maximum length."""
        long_query = "a" * (MAX_QUERY_LENGTH + 1)
        with pytest.raises(ValidationError):
            QueryInput(query=long_query)

    def test_query_at_maximum_length(self):
        """Test query at exactly maximum length."""
        max_query = "a" * MAX_QUERY_LENGTH
        query_input = QueryInput(query=max_query)
        assert len(query_input.query) == MAX_QUERY_LENGTH


class TestQueryInputXSSPrevention:
    """Test XSS prevention in QueryInput."""

    @pytest.mark.parametrize(
        "xss_attempt",
        [
            "<script>alert('xss')</script>",
            "<SCRIPT>malicious_code()</SCRIPT>",
            "<script src='evil.js'>",
            "<script type='text/javascript'>",
            "<ScRiPt>mixed_case()</ScRiPt>",
        ],
    )
    def test_script_tags_blocked(self, xss_attempt):
        """Test that script tags are blocked."""
        with pytest.raises(ValidationError) as exc_info:
            QueryInput(query=xss_attempt)
        assert "unsafe content" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "js_url",
        [
            "javascript:alert('xss')",
            "JAVASCRIPT:void(0)",
            "javascript:evil_function()",
        ],
    )
    def test_javascript_urls_blocked(self, js_url):
        """Test that JavaScript URLs are blocked."""
        with pytest.raises(ValidationError) as exc_info:
            QueryInput(query=js_url)
        assert "unsafe content" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "event_handler",
        [
            "onclick=alert('xss')",
            "onload =evil()",
            "onerror= hack()",
            "ONMOUSEOVER =track()",
        ],
    )
    def test_event_handlers_blocked(self, event_handler):
        """Test that event handlers are blocked."""
        with pytest.raises(ValidationError) as exc_info:
            QueryInput(query=event_handler)
        assert "unsafe content" in str(exc_info.value).lower()


class TestQueryInputTemplateInjection:
    """Test template injection prevention in QueryInput."""

    @pytest.mark.parametrize(
        "template_injection",
        [
            "{{user.password}}",
            "{{config.secret_key}}",
            "{{ system.admin }}",
            "{{__import__('os').system('whoami')}}",
        ],
    )
    def test_double_brace_templates_blocked(self, template_injection):
        """Test that double-brace template syntax is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            QueryInput(query=template_injection)
        assert "unsafe content" in str(exc_info.value).lower()

    @pytest.mark.parametrize(
        "template_literal",
        [
            "${process.env.SECRET}",
            "${require('child_process').exec('ls')}",
            "Hello ${name}",
        ],
    )
    def test_template_literals_blocked(self, template_literal):
        """Test that template literal syntax is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            QueryInput(query=template_literal)
        assert "unsafe content" in str(exc_info.value).lower()

    def test_legitimate_brackets_allowed(self):
        """Test that legitimate use of brackets is allowed."""
        # Square brackets are fine
        query = QueryInput(query="Array access like arr[0] is allowed")
        assert "[0]" in query.query

        # Single braces in code context
        query = QueryInput(query="JSON structure like {key: value} is ok")
        assert "{key:" in query.query


class TestQueryInputThreadId:
    """Test thread ID validation."""

    @pytest.mark.parametrize(
        "valid_thread_id",
        [
            "thread-123",
            "THREAD_456",
            "my-thread_id-789",
            "a",
            "123456789",
        ],
    )
    def test_valid_thread_ids(self, valid_thread_id):
        """Test valid thread ID formats."""
        query_input = QueryInput(query="test", thread_id=valid_thread_id)
        assert query_input.thread_id == valid_thread_id

    @pytest.mark.parametrize(
        "invalid_thread_id",
        [
            "thread/with/slashes",
            "thread\\with\\backslashes",
            "thread..traversal",
            "thread with spaces",
            "thread@email.com",
            "thread#hash",
        ],
    )
    def test_invalid_thread_ids(self, invalid_thread_id):
        """Test invalid thread ID formats are rejected."""
        with pytest.raises(ValidationError):
            QueryInput(query="test", thread_id=invalid_thread_id)

    def test_thread_id_max_length(self):
        """Test thread ID cannot exceed max length."""
        long_id = "a" * 101
        with pytest.raises(ValidationError):
            QueryInput(query="test", thread_id=long_id)

    def test_thread_id_at_max_length(self):
        """Test thread ID at exactly max length."""
        max_id = "a" * 100
        query_input = QueryInput(query="test", thread_id=max_id)
        assert len(query_input.thread_id) == 100


class TestMCTSConfigBounds:
    """Test MCTS configuration parameter bounds."""

    def test_default_values(self):
        """Test MCTS config default values."""
        config = MCTSConfig()
        assert config.iterations == 100
        assert config.exploration_weight == 1.414
        assert config.max_depth == 10
        assert config.simulation_timeout_seconds == 30.0

    @pytest.mark.parametrize("iterations", [1, 50, 100, 1000, 10000])
    def test_valid_iteration_counts(self, iterations):
        """Test valid iteration counts are accepted."""
        config = MCTSConfig(iterations=iterations)
        assert config.iterations == iterations

    def test_iterations_below_minimum(self):
        """Test iterations below minimum are rejected."""
        with pytest.raises(ValidationError):
            MCTSConfig(iterations=0)

    def test_iterations_above_maximum(self):
        """Test iterations above maximum are rejected."""
        with pytest.raises(ValidationError):
            MCTSConfig(iterations=MAX_ITERATIONS + 1)

    @pytest.mark.parametrize("weight", [0.0, 0.5, 1.0, 1.414, 2.0, 3.0, 10.0])
    def test_valid_exploration_weights(self, weight):
        """Test valid exploration weights are accepted."""
        config = MCTSConfig(exploration_weight=weight)
        assert config.exploration_weight == weight

    def test_exploration_weight_below_minimum(self):
        """Test exploration weight below minimum is rejected."""
        with pytest.raises(ValidationError):
            MCTSConfig(exploration_weight=-0.1)

    def test_exploration_weight_above_maximum(self):
        """Test exploration weight above maximum is rejected."""
        with pytest.raises(ValidationError):
            MCTSConfig(exploration_weight=10.1)

    @pytest.mark.parametrize("unusual_weight", [0.1, 0.4, 3.1, 5.0, 9.9])
    def test_unusual_exploration_weight_warning(self, unusual_weight):
        """Test that unusual exploration weights trigger warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MCTSConfig(exploration_weight=unusual_weight)
            assert len(w) == 1
            assert "outside typical range" in str(w[0].message)

    def test_max_depth_bounds(self):
        """Test max depth boundaries."""
        # Valid depths
        config = MCTSConfig(max_depth=1)
        assert config.max_depth == 1

        config = MCTSConfig(max_depth=50)
        assert config.max_depth == 50

        # Invalid depths
        with pytest.raises(ValidationError):
            MCTSConfig(max_depth=0)

        with pytest.raises(ValidationError):
            MCTSConfig(max_depth=51)

    def test_simulation_timeout_bounds(self):
        """Test simulation timeout boundaries."""
        # Valid timeouts
        config = MCTSConfig(simulation_timeout_seconds=1.0)
        assert config.simulation_timeout_seconds == 1.0

        config = MCTSConfig(simulation_timeout_seconds=300.0)
        assert config.simulation_timeout_seconds == 300.0

        # Invalid timeouts
        with pytest.raises(ValidationError):
            MCTSConfig(simulation_timeout_seconds=0.5)

        with pytest.raises(ValidationError):
            MCTSConfig(simulation_timeout_seconds=301.0)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError):
            MCTSConfig(iterations=100, unknown_field="value")


class TestAgentConfig:
    """Test AgentConfig validation."""

    def test_default_values(self):
        """Test AgentConfig defaults."""
        config = AgentConfig()
        assert config.max_iterations == 3
        assert config.consensus_threshold == 0.75
        assert config.temperature == 0.7
        assert config.max_tokens == 2048

    def test_temperature_bounds(self):
        """Test temperature parameter bounds."""
        config = AgentConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = AgentConfig(temperature=2.0)
        assert config.temperature == 2.0

        with pytest.raises(ValidationError):
            AgentConfig(temperature=-0.1)

        with pytest.raises(ValidationError):
            AgentConfig(temperature=2.1)

    def test_consensus_threshold_bounds(self):
        """Test consensus threshold bounds."""
        config = AgentConfig(consensus_threshold=0.0)
        assert config.consensus_threshold == 0.0

        config = AgentConfig(consensus_threshold=1.0)
        assert config.consensus_threshold == 1.0


class TestRAGConfig:
    """Test RAG configuration validation."""

    def test_default_values(self):
        """Test RAGConfig defaults."""
        config = RAGConfig()
        assert config.top_k == 5
        assert config.similarity_threshold == 0.5
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200

    def test_chunk_overlap_less_than_size(self):
        """Test chunk overlap must be less than size."""
        config = RAGConfig(chunk_size=1000, chunk_overlap=500)
        assert config.chunk_overlap < config.chunk_size

    def test_chunk_overlap_equals_size_rejected(self):
        """Test chunk overlap equal to size is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RAGConfig(chunk_size=1000, chunk_overlap=1000)
        assert "overlap must be less than chunk size" in str(exc_info.value)

    def test_chunk_overlap_greater_than_size_rejected(self):
        """Test chunk overlap greater than size is rejected."""
        with pytest.raises(ValidationError):
            RAGConfig(chunk_size=1000, chunk_overlap=1500)


class TestMCPToolInput:
    """Test MCP tool input validation."""

    @pytest.mark.parametrize(
        "valid_name",
        [
            "read_file",
            "search_documents",
            "toolName123",
            "my-tool_name",
        ],
    )
    def test_valid_tool_names(self, valid_name):
        """Test valid tool names are accepted."""
        tool_input = MCPToolInput(tool_name=valid_name)
        assert tool_input.tool_name == valid_name

    @pytest.mark.parametrize(
        "invalid_name",
        [
            "123tool",
            "-invalid",
            "_invalid",
            "tool/name",
            "tool\\name",
            "tool..name",
        ],
    )
    def test_invalid_tool_names_rejected(self, invalid_name):
        """Test invalid tool names are rejected."""
        with pytest.raises(ValidationError):
            MCPToolInput(tool_name=invalid_name)

    def test_parameters_validation(self):
        """Test tool parameters validation."""
        params = {
            "param1": "value1",
            "param2": 123,
            "param3": True,
        }
        tool_input = MCPToolInput(tool_name="test_tool", parameters=params)
        assert tool_input.parameters == params

    def test_too_many_parameters_rejected(self):
        """Test that too many parameters are rejected."""
        params = {f"param_{i}": i for i in range(51)}
        with pytest.raises(ValidationError) as exc_info:
            MCPToolInput(tool_name="test", parameters=params)
        assert "Too many parameters" in str(exc_info.value)

    def test_invalid_parameter_key_format(self):
        """Test invalid parameter key format is rejected."""
        params = {"123invalid": "value"}
        with pytest.raises(ValidationError) as exc_info:
            MCPToolInput(tool_name="test", parameters=params)
        assert "Invalid parameter key" in str(exc_info.value)


class TestFileReadInput:
    """Test FileReadInput validation for path security."""

    @pytest.mark.parametrize(
        "valid_path",
        [
            "data/file.txt",
            "documents/report.pdf",
            "src/main.py",
        ],
    )
    def test_valid_file_paths(self, valid_path):
        """Test valid file paths are accepted."""
        input_data = FileReadInput(file_path=valid_path)
        assert input_data.file_path == valid_path

    def test_path_traversal_blocked(self):
        """Test path traversal attempts are blocked."""
        with pytest.raises(ValidationError) as exc_info:
            FileReadInput(file_path="../../../etc/passwd")
        assert "Path traversal" in str(exc_info.value)

    @pytest.mark.parametrize(
        "restricted_path",
        [
            "/etc/shadow",
            "/root/.ssh/id_rsa",
            "~/.ssh/id_rsa",
            "/var/log/secure",
            "\\windows\\system32\\config",
        ],
    )
    def test_restricted_paths_blocked(self, restricted_path):
        """Test restricted system paths are blocked."""
        with pytest.raises(ValidationError) as exc_info:
            FileReadInput(file_path=restricted_path)
        assert "restricted directory" in str(exc_info.value)

    def test_absolute_path_warning(self):
        """Test absolute paths trigger warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            FileReadInput(file_path="/allowed/path/file.txt")
            assert len(w) == 1
            assert "Absolute file path" in str(w[0].message)


class TestWebFetchInput:
    """Test WebFetchInput URL validation."""

    @pytest.mark.parametrize(
        "valid_url",
        [
            "https://example.com",
            "https://api.example.com/data",
            "http://localhost:8080/api",
            "http://127.0.0.1:3000/test",
        ],
    )
    def test_valid_urls_accepted(self, valid_url):
        """Test valid URLs are accepted."""
        input_data = WebFetchInput(url=valid_url)
        assert input_data.url.strip() == valid_url

    def test_http_non_localhost_rejected(self):
        """Test non-localhost HTTP URLs are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            WebFetchInput(url="http://example.com")
        assert "HTTPS protocol" in str(exc_info.value)

    @pytest.mark.parametrize("invalid_char", ["<", ">", "'", '"', ";"])
    def test_invalid_characters_rejected(self, invalid_char):
        """Test URLs with invalid characters are rejected."""
        url = f"https://example.com/path{invalid_char}param"
        with pytest.raises(ValidationError) as exc_info:
            WebFetchInput(url=url)
        assert "invalid characters" in str(exc_info.value)


class TestBatchQueryInput:
    """Test batch query input validation."""

    def test_valid_batch(self):
        """Test valid batch of queries."""
        queries = [
            QueryInput(query="Query 1"),
            QueryInput(query="Query 2"),
        ]
        batch = BatchQueryInput(queries=queries)
        assert len(batch.queries) == 2

    def test_empty_batch_rejected(self):
        """Test empty batch is rejected."""
        with pytest.raises(ValidationError):
            BatchQueryInput(queries=[])

    def test_batch_exceeds_maximum_rejected(self):
        """Test batch exceeding maximum size is rejected."""
        queries = [QueryInput(query=f"Query {i}") for i in range(MAX_BATCH_SIZE + 1)]
        with pytest.raises(ValidationError) as exc_info:
            BatchQueryInput(queries=queries)
        error_msg = str(exc_info.value).lower()
        assert "too_long" in error_msg or "at most" in error_msg


class TestAPIRequestMetadata:
    """Test API request metadata validation."""

    def test_valid_request_id(self):
        """Test valid request ID."""
        metadata = APIRequestMetadata(request_id="req-123-abc")
        assert metadata.request_id == "req-123-abc"

    @pytest.mark.parametrize(
        "valid_ip",
        [
            "192.168.1.1",
            "10.0.0.1",
            "::1",
            "2001:db8::1",
        ],
    )
    def test_valid_ip_addresses(self, valid_ip):
        """Test valid IP addresses."""
        metadata = APIRequestMetadata(request_id="test", source_ip=valid_ip)
        assert metadata.source_ip == valid_ip

    def test_invalid_ip_rejected(self):
        """Test invalid IP address is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            APIRequestMetadata(request_id="test", source_ip="not.an.ip")
        assert "Invalid IP address" in str(exc_info.value)


class TestConvenienceFunctions:
    """Test convenience validation functions."""

    def test_validate_query_function(self):
        """Test validate_query convenience function."""
        result = validate_query("Test query", use_mcts=True)
        assert isinstance(result, QueryInput)
        assert result.query == "Test query"
        assert result.use_mcts is True

    def test_validate_mcts_config_function(self):
        """Test validate_mcts_config convenience function."""
        result = validate_mcts_config(iterations=200, exploration_weight=2.0)
        assert isinstance(result, MCTSConfig)
        assert result.iterations == 200
        assert result.exploration_weight == 2.0

    def test_validate_tool_input_function(self):
        """Test validate_tool_input convenience function."""
        result = validate_tool_input("test_tool", {"param": "value"})
        assert isinstance(result, MCPToolInput)
        assert result.tool_name == "test_tool"
        assert result.parameters == {"param": "value"}


class TestSettingsInitialization:
    """Test Settings initialization and configuration."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def teardown_method(self):
        """Clean up after each test."""
        reset_settings()

    def test_lmstudio_provider_defaults(self):
        """Test LMStudio provider with default URL."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            assert settings.LLM_PROVIDER == LLMProvider.LMSTUDIO
            # Allow either localhost or 127.0.0.1
            assert settings.LMSTUDIO_BASE_URL in ["http://localhost:1234/v1", "http://127.0.0.1:1234/v1"]

    def test_mcts_default_values(self):
        """Test MCTS default configuration values."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            assert settings.MCTS_ITERATIONS == 100
            assert settings.MCTS_C == 1.414

    def test_log_level_default(self):
        """Test default log level."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            assert settings.LOG_LEVEL == LogLevel.INFO


class TestSettingsEnvironmentVariables:
    """Test environment variable loading in Settings."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def teardown_method(self):
        """Clean up after each test."""
        reset_settings()

    def test_mcts_iterations_from_env(self):
        """Test MCTS iterations loaded from environment."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
                "MCTS_ITERATIONS": "500",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            assert settings.MCTS_ITERATIONS == 500

    def test_mcts_c_from_env(self):
        """Test MCTS_C loaded from environment."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
                "MCTS_C": "2.0",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            assert settings.MCTS_C == 2.0

    def test_log_level_from_env(self):
        """Test log level from environment variable."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
                "LOG_LEVEL": "DEBUG",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            assert settings.LOG_LEVEL == LogLevel.DEBUG

    def test_invalid_mcts_iterations_rejected(self):
        """Test invalid MCTS iterations are rejected."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
                "MCTS_ITERATIONS": "20000",  # Exceeds maximum
            },
            clear=False,
        ):
            reset_settings()
            with pytest.raises(ValidationError):
                Settings()

    def test_invalid_mcts_c_rejected(self):
        """Test invalid MCTS_C is rejected."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
                "MCTS_C": "15.0",  # Exceeds maximum of 10.0
            },
            clear=False,
        ):
            reset_settings()
            with pytest.raises(ValidationError):
                Settings()


class TestProviderEnumValidation:
    """Test LLM provider enum validation."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def teardown_method(self):
        """Clean up after each test."""
        reset_settings()

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "lmstudio"])
    def test_valid_providers(self, provider):
        """Test valid provider enum values."""
        enum_value = LLMProvider(provider)
        assert enum_value.value == provider

    def test_invalid_provider_rejected(self):
        """Test invalid provider value is rejected."""
        with pytest.raises(ValueError):
            LLMProvider("invalid_provider")

    def test_openai_requires_api_key(self):
        """Test OpenAI provider requires API key."""
        # Create environment without OpenAI key - need to patch _env_file to prevent loading from .env

        env_without_key = {
            "LLM_PROVIDER": "openai",
        }
        # Explicitly set to None by removing the key entirely and disable .env loading
        with patch.dict(os.environ, env_without_key, clear=True):
            reset_settings()

            # Create a Settings class that doesn't load from .env file
            class TestSettings(Settings):
                model_config = Settings.model_config.copy()

            TestSettings.model_config["env_file"] = None

            with pytest.raises(ValidationError) as exc_info:
                TestSettings()
            assert "OPENAI_API_KEY is required" in str(exc_info.value)

    def test_anthropic_requires_api_key(self):
        """Test Anthropic provider requires API key."""
        # Create environment without Anthropic key
        env_without_key = {
            "LLM_PROVIDER": "anthropic",
        }
        with patch.dict(os.environ, env_without_key, clear=True):
            reset_settings()

            # Create a Settings class that doesn't load from .env file
            class TestSettings(Settings):
                model_config = Settings.model_config.copy()

            TestSettings.model_config["env_file"] = None

            with pytest.raises(ValidationError) as exc_info:
                TestSettings()
            assert "ANTHROPIC_API_KEY is required" in str(exc_info.value)


class TestSecretStrHandling:
    """Test SecretStr handling for API keys."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def teardown_method(self):
        """Clean up after each test."""
        reset_settings()

    def test_openai_key_is_secret(self):
        """Test OpenAI API key is stored as SecretStr."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-test123456789012345678901234567890",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            assert isinstance(settings.OPENAI_API_KEY, SecretStr)
            assert settings.OPENAI_API_KEY.get_secret_value() == "sk-test123456789012345678901234567890"

    def test_anthropic_key_is_secret(self):
        """Test Anthropic API key is stored as SecretStr."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": "sk-ant-test1234567890123456",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            assert isinstance(settings.ANTHROPIC_API_KEY, SecretStr)

    def test_openai_key_format_validation(self):
        """Test OpenAI API key format is validated."""
        # Invalid: doesn't start with sk-
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "invalid_key_format_12345678901234567890",
            },
            clear=False,
        ):
            reset_settings()
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "should start with 'sk-'" in str(exc_info.value)

    def test_openai_key_placeholder_rejected(self):
        """Test OpenAI placeholder keys are rejected."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-xxx",
            },
            clear=False,
        ):
            reset_settings()
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "placeholder" in str(exc_info.value)

    def test_api_key_too_short_rejected(self):
        """Test API keys that are too short are rejected."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-short",
            },
            clear=False,
        ):
            reset_settings()
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "too short" in str(exc_info.value)


class TestSecretMasking:
    """Test secret masking in logs and representations."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def teardown_method(self):
        """Clean up after each test."""
        reset_settings()

    def test_safe_dict_masks_secrets(self):
        """Test safe_dict masks all API keys."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-real_secret_key_12345678901234567890",
                "ANTHROPIC_API_KEY": "sk-ant-another_secret_1234567890",
                "BRAINTRUST_API_KEY": "br-secret123456789012345678",
                "PINECONE_API_KEY": "pc-secret1234567890123456789",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            safe_data = settings.safe_dict()

            assert safe_data["OPENAI_API_KEY"] == "***MASKED***"
            assert safe_data["ANTHROPIC_API_KEY"] == "***MASKED***"
            assert safe_data["BRAINTRUST_API_KEY"] == "***MASKED***"
            assert safe_data["PINECONE_API_KEY"] == "***MASKED***"

            # Ensure actual keys are not in safe dict
            assert "sk-real_secret_key" not in str(safe_data)
            assert "sk-ant-another_secret" not in str(safe_data)

    def test_repr_does_not_expose_secrets(self):
        """Test __repr__ does not expose secrets."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-super_secret_key_123456789012345678901234567890",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            repr_str = repr(settings)

            assert "sk-super_secret_key" not in repr_str
            assert "LLM_PROVIDER" in repr_str
            assert "LOG_LEVEL" in repr_str

    def test_get_api_key_returns_value(self):
        """Test get_api_key returns actual secret value."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-test_api_key_1234567890123456789012345678901234567890",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            key = settings.get_api_key()

            assert key == "sk-test_api_key_1234567890123456789012345678901234567890"

    def test_get_api_key_for_anthropic(self):
        """Test get_api_key returns Anthropic key when that provider is selected."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": "sk-ant-test1234567890123456",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            key = settings.get_api_key()

            assert key == "sk-ant-test1234567890123456"

    def test_get_api_key_for_lmstudio_returns_none(self):
        """Test get_api_key returns None for LMStudio."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            key = settings.get_api_key()

            assert key is None


class TestSettingsValidation:
    """Test additional settings validation."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def teardown_method(self):
        """Clean up after each test."""
        reset_settings()

    def test_pinecone_host_must_be_https(self):
        """Test Pinecone host must use HTTPS."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
                "PINECONE_HOST": "http://invalid.pinecone.io",
            },
            clear=False,
        ):
            reset_settings()
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "must start with https://" in str(exc_info.value)

    def test_pinecone_host_must_be_pinecone_domain(self):
        """Test Pinecone host must be pinecone.io domain."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
                "PINECONE_HOST": "https://not-pinecone.com",
            },
            clear=False,
        ):
            reset_settings()
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "pinecone.io" in str(exc_info.value)

    def test_lmstudio_url_validation(self):
        """Test LMStudio URL validation."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
                "LMSTUDIO_BASE_URL": "invalid_url",
            },
            clear=False,
        ):
            reset_settings()
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "must start with http://" in str(exc_info.value)

    def test_lmstudio_non_localhost_warning(self):
        """Test LMStudio non-localhost URL triggers warning."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
                "LMSTUDIO_BASE_URL": "http://remote-server.com:1234/v1",
            },
            clear=False,
        ):
            reset_settings()
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                Settings()
                assert len(w) == 1
                assert "non-localhost" in str(w[0].message)

    def test_s3_bucket_name_validation(self):
        """Test S3 bucket name validation."""
        # Valid bucket name
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
                "S3_BUCKET": "my-valid-bucket-123",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            assert settings.S3_BUCKET == "my-valid-bucket-123"

    def test_s3_bucket_invalid_characters_rejected(self):
        """Test S3 bucket with invalid characters is rejected."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
                "S3_BUCKET": "invalid_bucket_name!",
            },
            clear=False,
        ):
            reset_settings()
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "lowercase letters" in str(exc_info.value)

    def test_s3_bucket_too_short_rejected(self):
        """Test S3 bucket name that is too short is rejected."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
                "S3_BUCKET": "ab",
            },
            clear=False,
        ):
            reset_settings()
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "3-63 characters" in str(exc_info.value)

    def test_otel_endpoint_validation(self):
        """Test OpenTelemetry endpoint URL validation."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "invalid://endpoint",
            },
            clear=False,
        ):
            reset_settings()
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "must start with http://" in str(exc_info.value)


class TestGlobalSettingsFunctions:
    """Test global settings instance management."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def teardown_method(self):
        """Clean up after each test."""
        reset_settings()

    def test_get_settings_returns_singleton(self):
        """Test get_settings returns same instance."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
            },
            clear=False,
        ):
            reset_settings()
            settings1 = get_settings()
            settings2 = get_settings()
            assert settings1 is settings2

    def test_reset_settings_clears_instance(self):
        """Test reset_settings clears the cached instance."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
            },
            clear=False,
        ):
            reset_settings()
            settings1 = get_settings()
            reset_settings()
            settings2 = get_settings()
            # Should be different instances after reset
            assert settings1 is not settings2


class TestTypeCoercion:
    """Test type coercion for configuration values."""

    def setup_method(self):
        """Reset settings before each test."""
        reset_settings()

    def teardown_method(self):
        """Clean up after each test."""
        reset_settings()

    def test_string_to_int_coercion(self):
        """Test string environment variables are coerced to int."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
                "MCTS_ITERATIONS": "250",
                "HTTP_TIMEOUT_SECONDS": "60",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            assert settings.MCTS_ITERATIONS == 250
            assert isinstance(settings.MCTS_ITERATIONS, int)
            assert settings.HTTP_TIMEOUT_SECONDS == 60
            assert isinstance(settings.HTTP_TIMEOUT_SECONDS, int)

    def test_string_to_float_coercion(self):
        """Test string environment variables are coerced to float."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "lmstudio",
                "MCTS_C": "2.5",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            assert settings.MCTS_C == 2.5
            assert isinstance(settings.MCTS_C, float)

    def test_string_to_enum_coercion(self):
        """Test string environment variables are coerced to enum."""
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "anthropic",
                "ANTHROPIC_API_KEY": "sk-ant-valid_key_12345678901234567890",
                "LOG_LEVEL": "WARNING",
            },
            clear=False,
        ):
            reset_settings()
            settings = Settings()
            assert settings.LLM_PROVIDER == LLMProvider.ANTHROPIC
            assert isinstance(settings.LLM_PROVIDER, LLMProvider)
            assert settings.LOG_LEVEL == LogLevel.WARNING
            assert isinstance(settings.LOG_LEVEL, LogLevel)
