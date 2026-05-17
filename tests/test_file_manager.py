from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from intent_classifier.file_manager import (
    CommandMemory,
    CommandResult,
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
        self.calls = 0

    def classify(self, question: str) -> FileCommand:
        self.calls += 1
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
    return FileManagerAgent(
        tmp_path,
        classifier=StaticClassifier(command),
        allow_risky=allow_risky,
        memory=memory_for(tmp_path),
    )


def memory_for(tmp_path: Path) -> CommandMemory:
    return CommandMemory(tmp_path.parent / f"{tmp_path.name}-filemanager-cache")


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


def test_repeated_identical_request_uses_cached_command(tmp_path: Path) -> None:
    memory = memory_for(tmp_path)
    classifier = StaticClassifier(FileCommand("create_file", {"target": "notes.txt"}))
    agent = FileManagerAgent(tmp_path, classifier=classifier, memory=memory)

    first = agent.run("please create a file named notes.txt")
    (tmp_path / "notes.txt").unlink()
    second = agent.run("please create a file named notes.txt")

    assert first.success is True
    assert second.success is True
    assert classifier.calls == 1
    assert (tmp_path / "notes.txt").is_file()


def test_no_cache_bypasses_cached_command(tmp_path: Path) -> None:
    memory = memory_for(tmp_path)
    memory.cache_command("create a file", FileCommand("create_file", {"target": "cached.txt"}))
    classifier = StaticClassifier(FileCommand("create_file", {"target": "fresh.txt"}))
    agent = FileManagerAgent(tmp_path, classifier=classifier, memory=memory, use_cache=False)

    result = agent.run("create a file")

    assert result.success is True
    assert classifier.calls == 1
    assert not (tmp_path / "cached.txt").exists()
    assert (tmp_path / "fresh.txt").is_file()


def test_risky_and_delete_classifications_are_not_reused_from_cache(tmp_path: Path) -> None:
    memory = memory_for(tmp_path)
    memory.cache_command(
        "overwrite notes",
        FileCommand("write_file", {"target": "notes.txt", "content": "new"}, requires_confirmation=True),
    )
    memory.cache_command("delete notes", FileCommand("delete_path", {"target": "notes.txt"}))

    assert memory.cached_command("overwrite notes") is None
    assert memory.cached_command("delete notes") is None


def test_history_resolves_rename_it_without_llm(tmp_path: Path) -> None:
    memory = memory_for(tmp_path)
    classifier = StaticClassifier(FileCommand("create_directory", {"target": "goal"}))
    agent = FileManagerAgent(tmp_path, classifier=classifier, memory=memory)

    created = agent.run("create a directory named goal")
    renamed = agent.run("rename it to archive")

    assert created.success is True
    assert renamed.success is True
    assert classifier.calls == 1
    assert not (tmp_path / "goal").exists()
    assert (tmp_path / "archive").is_dir()


def test_history_resolves_read_it_without_llm(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello")
    memory = memory_for(tmp_path)
    classifier = StaticClassifier(FileCommand("read_file", {"target": "notes.txt"}))
    agent = FileManagerAgent(tmp_path, classifier=classifier, memory=memory)

    first = agent.run("read notes.txt")
    second = agent.run("read it")

    assert first.message == "hello"
    assert second.message == "hello"
    assert classifier.calls == 1


def test_ambiguous_follow_up_returns_clarification_without_llm(tmp_path: Path) -> None:
    memory = memory_for(tmp_path)
    classifier = StaticClassifier(FileCommand("create_file", {"target": "should-not-run.txt"}))
    agent = FileManagerAgent(tmp_path, classifier=classifier, memory=memory)

    result = agent.run("rename it")

    assert result.success is False
    assert result.intent == "clarification_required"
    assert classifier.calls == 0
    assert not (tmp_path / "should-not-run.txt").exists()


def test_history_resolves_delete_it_but_requires_confirmation(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("hello")
    memory = memory_for(tmp_path)
    classifier = StaticClassifier(FileCommand("read_file", {"target": "notes.txt"}))
    agent = FileManagerAgent(tmp_path, classifier=classifier, memory=memory)

    agent.run("read notes.txt")
    result = agent.run("delete it")

    assert result.success is False
    assert result.requires_confirmation is True
    assert classifier.calls == 1
    assert (tmp_path / "notes.txt").exists()


def test_cached_path_escape_is_still_rejected(tmp_path: Path) -> None:
    memory = memory_for(tmp_path)
    memory.cache_command("make outside", FileCommand("create_file", {"target": "../outside.txt"}))
    classifier = StaticClassifier(FileCommand("create_file", {"target": "should-not-run.txt"}))
    agent = FileManagerAgent(tmp_path, classifier=classifier, memory=memory)

    with pytest.raises(ValueError):
        agent.run("make outside")

    assert classifier.calls == 0


def test_history_path_escape_is_still_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside")
    memory = memory_for(tmp_path)
    memory.cache_dir.mkdir(parents=True, exist_ok=True)
    memory.action_history_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-17T00:00:00+00:00",
                "question": "malicious history",
                "root": str(tmp_path),
                "intent": "read_file",
                "params": {"target": "../outside.txt"},
                "requires_confirmation": False,
                "path": "../outside.txt",
                "message": "outside",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    classifier = StaticClassifier(FileCommand("read_file", {"target": "should-not-run.txt"}))
    agent = FileManagerAgent(tmp_path, classifier=classifier, memory=memory)

    with pytest.raises(ValueError):
        agent.run("read it")

    assert classifier.calls == 0


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
    memory_dir = tmp_path / ".filemanager-cache"
    original_memory = CommandMemory

    class CliClassifier:
        def classify(self, question: str) -> FileCommand:
            return FileCommand("delete_path", {"target": "old.txt"}, requires_confirmation=True)

    monkeypatch.setattr(
        "intent_classifier.file_manager.LLMIntentClassifier",
        lambda: CliClassifier(),
    )
    monkeypatch.setattr(
        "intent_classifier.file_manager.CommandMemory",
        lambda: original_memory(memory_dir),
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
    original_memory = CommandMemory
    monkeypatch.setattr(
        "intent_classifier.file_manager.CommandMemory",
        lambda: original_memory(tmp_path / ".filemanager-cache"),
    )
    monkeypatch.setattr("sys.argv", ["filemanager", "--root", str(tmp_path), "list files"])

    assert main() == 1
    assert "OPENAI_API_KEY is required" in capsys.readouterr().out


def test_cli_clear_cache_removes_memory_files(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / ".filemanager-cache"
    memory = CommandMemory(memory_dir)
    memory.cache_command("create note", FileCommand("create_file", {"target": "notes.txt"}))
    memory.append_history(
        "create note",
        FileCommand("create_file", {"target": "notes.txt"}),
        result=CommandResult(
            True,
            "Created file: notes.txt",
            "create_file",
            tmp_path / "notes.txt",
        ),
        root=tmp_path,
    )
    original_memory = CommandMemory
    monkeypatch.setattr(
        "intent_classifier.file_manager.CommandMemory",
        lambda: original_memory(memory_dir),
    )
    monkeypatch.setattr("sys.argv", ["filemanager", "--clear-cache"])

    assert main() == 0
    assert "Cleared filemanager cache and history." in capsys.readouterr().out
    assert not memory.classification_cache_path.exists()
    assert not memory.action_history_path.exists()


def test_cli_history_prints_recent_actions(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    memory_dir = tmp_path / ".filemanager-cache"
    memory = CommandMemory(memory_dir)
    target = tmp_path / "notes.txt"
    target.write_text("hello")
    memory.append_history(
        "read notes",
        FileCommand("read_file", {"target": "notes.txt"}),
        result=CommandResult(True, "hello", "read_file", target),
        root=tmp_path,
    )
    original_memory = CommandMemory
    monkeypatch.setattr(
        "intent_classifier.file_manager.CommandMemory",
        lambda: original_memory(memory_dir),
    )
    monkeypatch.setattr("sys.argv", ["filemanager", "--history"])

    assert main() == 0
    output = capsys.readouterr().out
    assert "read_file notes.txt :: read notes" in output
