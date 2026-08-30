import base64
from unittest.mock import mock_open, patch

import pytest

from music_evalkit.data.schema import Instrument, MCQInference, OpenEndedInference, PairwiseInference
from music_evalkit.messages.message_builder import MessageBuilder


@pytest.fixture
def sample_open_ended() -> OpenEndedInference:
    """Create a sample open-ended evaluation sample."""
    return OpenEndedInference(
        id="test-001",
        instruction="Play C major scale ascending",
        instrument=Instrument.PIANO,
        piece_type="exercise",
        audio_setting="single",
        mistake_category="pitch",
        mistake="The 4th note was played as F# instead of F natural",
        feedback="Focus on the F natural in the C major scale",
        audio_path="student.wav",
    )


@pytest.fixture
def sample_mcq() -> MCQInference:
    """Create a sample MCQ evaluation sample."""
    return MCQInference(
        id="test-002",
        instruction="Play C major chord",
        instrument=Instrument.GUITAR,
        piece_type="exercise",
        audio_setting="single",
        mistake_category="pitch",
        options="pitch, rhythm, dynamics, no_mistake",
        answer="pitch",
        audio_path="student.wav",
    )


@pytest.fixture
def sample_pairwise() -> PairwiseInference:
    """Create a sample pairwise evaluation sample."""
    return PairwiseInference(
        id="test-003",
        question="Which recording is closer to 80 BPM?",
        instrument=Instrument.PIANO,
        category="tempo",
        label="A",
        audio_path="concat.wav",
    )


class TestMessageBuilderBuild:
    """Tests for MessageBuilder.build() method."""

    @patch("builtins.open", mock_open(read_data=b"fake audio data"))
    def test_build_returns_system_and_user_messages(self, sample_open_ended):
        builder = MessageBuilder()
        messages = builder.build(sample_open_ended)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @patch("builtins.open", mock_open(read_data=b"fake audio data"))
    def test_build_user_message_structure(self, sample_open_ended):
        builder = MessageBuilder()
        messages = builder.build(sample_open_ended)

        user_content = messages[1]["content"]
        assert isinstance(user_content, list)
        assert len(user_content) == 2  # text + audio

        assert user_content[0]["type"] == "text"
        assert "text" in user_content[0]

        assert user_content[1]["type"] == "input_audio"
        assert "input_audio" in user_content[1]

    @patch("builtins.open", mock_open(read_data=b"fake audio data"))
    def test_build_mcq_sample(self, sample_mcq):
        builder = MessageBuilder()
        messages = builder.build(sample_mcq)

        text_content = messages[1]["content"][0]["text"]

        # MCQ prompt should include the options as labeled choices
        assert "A) pitch" in text_content
        assert "B) rhythm" in text_content
        assert "C) dynamics" in text_content
        assert "D) no_mistake" in text_content

    @patch("builtins.open", mock_open(read_data=b"fake audio data"))
    def test_build_pairwise_sample(self, sample_pairwise):
        builder = MessageBuilder()
        messages = builder.build(sample_pairwise)

        text_content = messages[1]["content"][0]["text"]
        assert "Which recording is closer to 80 BPM?" in text_content

    @patch("builtins.open", mock_open(read_data=b"fake audio data"))
    def test_build_task_type_override(self, sample_mcq):
        """An explicit task_type should be used instead of auto-detection."""
        from music_evalkit.data.schema import TaskType

        builder = MessageBuilder()
        messages_auto = builder.build(sample_mcq)
        messages_explicit = builder.build(sample_mcq, task_type=TaskType.MCQ)

        assert messages_auto[0]["content"] == messages_explicit[0]["content"]


class TestTaskTypeDetection:
    """Tests for automatic task type detection from sample type."""

    @patch("builtins.open", mock_open(read_data=b"fake audio data"))
    def test_detects_open_ended(self, sample_open_ended):
        builder = MessageBuilder()
        assert builder._get_task_type(sample_open_ended).value == "open_ended"

    @patch("builtins.open", mock_open(read_data=b"fake audio data"))
    def test_detects_mcq(self, sample_mcq):
        builder = MessageBuilder()
        assert builder._get_task_type(sample_mcq).value == "mcq"

    @patch("builtins.open", mock_open(read_data=b"fake audio data"))
    def test_detects_pairwise(self, sample_pairwise):
        builder = MessageBuilder()
        assert builder._get_task_type(sample_pairwise).value == "pairwise"


class TestCreateAudioContent:
    """Tests for _create_audio_content method."""

    def test_create_audio_content_structure(self):
        from pathlib import Path

        builder = MessageBuilder()
        test_data = b"audio bytes"

        with patch("builtins.open", mock_open(read_data=test_data)):
            content = builder._create_audio_content(Path("test.wav"))

        assert content["type"] == "input_audio"
        assert content["input_audio"]["format"] == "wav"
        assert content["input_audio"]["data"] == base64.b64encode(test_data).decode("utf-8")

    def test_create_audio_content_mp3(self):
        from pathlib import Path

        builder = MessageBuilder()
        test_data = b"mp3 audio data"

        with patch("builtins.open", mock_open(read_data=test_data)):
            content = builder._create_audio_content(Path("song.mp3"))

        assert content["input_audio"]["format"] == "mp3"
