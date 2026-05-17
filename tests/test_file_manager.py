from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from intent_classifier.file_manager import (
    DEFAULT_MODEL,
    FileCommand,
    FileManagerAgent,
    LLMFileCommand,
    LLMIntentClassifier,
    main,
)


class StaticClassifier:
    def __init__(self, command: FileCommand) -> None:
        self.command = command

    def classify(self, question: str) -> FileCommand:
        return self.command


class StubResponses:
    def __init__(self, parsed: LLMFileCommand) -> None:
        self.parsed = parsed
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_parsed=self.parsed)


class StubClient:
    def __init__(self, parsed: LLMFileCommand) -> None:
        self.responses = StubResponses(parsed)


def agent_for(command: FileCommand, tmp_path: Path, allow_risky: bool = False) -> FileManagerAgent:
    return FileManagerAgent(tmp_path, classifier=StaticClassifier(command), allow_risky=allow_risky)


def test_llm_classifier_creates_directory_from_desire_style_question() -> None:
    client = StubClient(LLMFileCommand(intent="create_directory", target="goal"))
    classifier = LLMIntentClassifier(client=client)

    command = classifier.classify(
        "I am working on a new project for which I would like to have a new directory named goal"
    )

    assert command.intent == "create_directory"
    assert command.params == {"target": "goal"}
    assert client.responses.calls[0]["model"] == DEFAULT_MODEL


def test_llm_classifier_renames_file() -> None:
    client = StubClient(
        LLMFileCommand(intent="rename_path", source="notes.txt", destination="todo.txt")
    )
    classifier = LLMIntentClassifier(client=client)

    command = classifier.classify("rename my notes file to todo.txt")

    assert command.intent == "rename_path"
    assert command.params == {"source": "notes.txt", "destination": "todo.txt"}


def test_llm_classifier_returns_clarification() -> None:
    client = StubClient(
        LLMFileCommand(
            intent="clarification_required",
            clarification_question="Which files should I clean up?",
        )
    )
    classifier = LLMIntentClassifier(client=client)

    command = classifier.classify("clean this up")

    assert command.intent == "clarification_required"
    assert command.clarification_question == "Which files should I clean up?"


def test_creates_directory_from_classified_command(tmp_path: Path) -> None:
    agent = agent_for(FileCommand("create_directory", {"target": "goal"}), tmp_path)

    result = agent.run("ignored by static classifier")

    assert result.success is True
    assert result.intent == "create_directory"
    assert (tmp_path / "goal").is_dir()
    assert result.message == "Created directory: goal"


def test_creates_file_from_classified_command(tmp_path: Path) -> None:
    agent = agent_for(FileCommand("create_file", {"target": "notes.txt"}), tmp_path)

    result = agent.run("ignored by static classifier")

    assert result.success is True
    assert result.intent == "create_file"
    assert (tmp_path / "notes.txt").is_file()


def test_lists_current_directory(tmp_path: Path) -> None:
    (tmp_path / "b.txt").touch()
    (tmp_path / "a.txt").touch()
    agent = agent_for(FileCommand("list_directory"), tmp_path)

    result = agent.run("ignored by static classifier")

    assert result.success is True
    assert result.intent == "list_directory"
    assert result.message.splitlines() == ["a.txt", "b.txt"]


def test_renames_file_from_classified_command(tmp_path: Path) -> None:
    (tmp_path / "old.txt").touch()
    agent = agent_for(
        FileCommand("rename_path", {"source": "old.txt", "destination": "new.txt"}), tmp_path
    )

    result = agent.run("ignored by static classifier")

    assert result.success is True
    assert result.intent == "rename_path"
    assert not (tmp_path / "old.txt").exists()
    assert (tmp_path / "new.txt").is_file()
    assert result.message == "Renamed old.txt to new.txt"


def test_copies_file_from_classified_command(tmp_path: Path) -> None:
    (tmp_path / "source.txt").write_text("hello")
    agent = agent_for(
        FileCommand("copy_path", {"source": "source.txt", "destination": "copy.txt"}), tmp_path
    )

    result = agent.run("ignored by static classifier")

    assert result.success is True
    assert (tmp_path / "source.txt").read_text() == "hello"
    assert (tmp_path / "copy.txt").read_text() == "hello"


def test_moves_file_from_classified_command(tmp_path: Path) -> None:
    (tmp_path / "source.txt").write_text("hello")
    agent = agent_for(
        FileCommand("move_path", {"source": "source.txt", "destination": "moved.txt"}), tmp_path
    )

    result = agent.run("ignored by static classifier")

    assert result.success is True
    assert not (tmp_path / "source.txt").exists()
    assert (tmp_path / "moved.txt").read_text() == "hello"


def test_reads_file_from_classified_command(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello")
    agent = agent_for(FileCommand("read_file", {"target": "notes.txt"}), tmp_path)

    result = agent.run("ignored by static classifier")

    assert result.success is True
    assert result.message == "hello"


def test_writes_file_from_classified_command(tmp_path: Path) -> None:
    agent = agent_for(FileCommand("write_file", {"target": "notes.txt", "content": "hello"}), tmp_path)

    result = agent.run("ignored by static classifier")

    assert result.success is True
    assert (tmp_path / "notes.txt").read_text() == "hello"


@pytest.mark.parametrize(
    "command",
    [
        FileCommand("create_directory", {"target": "../outside"}),
        FileCommand("create_file", {"target": "/tmp/outside.txt"}),
        FileCommand("rename_path", {"source": "../outside", "destination": "inside.txt"}),
    ],
)
def test_rejects_paths_outside_working_directory(tmp_path: Path, command: FileCommand) -> None:
    agent = agent_for(command, tmp_path)

    with pytest.raises(ValueError):
        agent.run("ignored by static classifier")


def test_blocks_overwrite_without_confirmation(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("old")
    agent = agent_for(
        FileCommand("write_file", {"target": "notes.txt", "content": "new"}, requires_confirmation=True),
        tmp_path,
    )

    result = agent.run("ignored by static classifier")

    assert result.success is False
    assert result.requires_confirmation is True
    assert (tmp_path / "notes.txt").read_text() == "old"


def test_yes_allows_overwrite(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("old")
    agent = agent_for(
        FileCommand("write_file", {"target": "notes.txt", "content": "new"}, requires_confirmation=True),
        tmp_path,
        allow_risky=True,
    )

    result = agent.run("ignored by static classifier")

    assert result.success is True
    assert (tmp_path / "notes.txt").read_text() == "new"


def test_delete_requires_confirmation(tmp_path: Path) -> None:
    (tmp_path / "old.txt").touch()
    agent = agent_for(FileCommand("delete_path", {"target": "old.txt"}, requires_confirmation=True), tmp_path)

    result = agent.run("ignored by static classifier")

    assert result.success is False
    assert result.requires_confirmation is True
    assert (tmp_path / "old.txt").exists()


def test_requires_confirmation_flag_blocks_mutating_operations(tmp_path: Path) -> None:
    agent = agent_for(
        FileCommand("create_file", {"target": "generated.txt"}, requires_confirmation=True),
        tmp_path,
    )

    result = agent.run("ignored by static classifier")

    assert result.success is False
    assert result.requires_confirmation is True
    assert not (tmp_path / "generated.txt").exists()


def test_recursive_copy_requires_confirmation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "notes.txt").write_text("hello")
    agent = agent_for(FileCommand("copy_path", {"source": "source", "destination": "copy"}), tmp_path)

    result = agent.run("ignored by static classifier")

    assert result.success is False
    assert result.requires_confirmation is True
    assert not (tmp_path / "copy").exists()


def test_yes_allows_delete(tmp_path: Path) -> None:
    (tmp_path / "old.txt").touch()
    agent = agent_for(
        FileCommand("delete_path", {"target": "old.txt"}, requires_confirmation=True),
        tmp_path,
        allow_risky=True,
    )

    result = agent.run("ignored by static classifier")

    assert result.success is True
    assert not (tmp_path / "old.txt").exists()


def test_missing_openai_api_key_returns_clarification(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = FileManagerAgent(tmp_path)

    result = agent.run("create a file named notes.txt")

    assert result.success is False
    assert result.intent == "clarification_required"
    assert "OPENAI_API_KEY is required" in result.message


def test_cli_yes_allows_confirmed_delete(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    target = tmp_path / "old.txt"
    target.touch()

    class CliClassifier:
        def classify(self, question: str) -> FileCommand:
            return FileCommand("delete_path", {"target": "old.txt"}, requires_confirmation=True)

    monkeypatch.setattr(
        "intent_classifier.file_manager.LLMIntentClassifier",
        lambda: CliClassifier(),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["filemanager", "--root", str(tmp_path), "--yes", "delete old.txt"],
    )

    assert main() == 0
    assert not target.exists()
    assert "Deleted: old.txt" in capsys.readouterr().out


def test_cli_missing_openai_api_key_returns_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["filemanager", "--root", str(tmp_path), "list files"])

    assert main() == 1
    assert "OPENAI_API_KEY is required" in capsys.readouterr().out
