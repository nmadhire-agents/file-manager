"""A local filemanager agent with LLM-based intent classification."""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict


ALLOWED_INTENTS = (
    "create_directory",
    "create_file",
    "list_directory",
    "rename_path",
    "copy_path",
    "move_path",
    "delete_path",
    "read_file",
    "write_file",
    "unknown",
    "clarification_required",
)

DEFAULT_MODEL = "gpt-5.4-nano"

CLASSIFIER_INSTRUCTIONS = """You classify file-management requests for a local CLI agent.
Return only the structured command. Do not execute anything.

Allowed intents:
- create_directory: create one directory at target.
- create_file: create one empty file at target.
- list_directory: list the current directory or target directory.
- rename_path: rename source to destination.
- copy_path: copy source to destination.
- move_path: move source to destination.
- delete_path: delete target.
- read_file: read target file.
- write_file: write content to target file.
- unknown: request is not a file-management action.
- clarification_required: file action is likely, but required details are missing.

Paths must be relative paths exactly as the user named them. Never invent filenames.
Set requires_confirmation to true for delete, overwrite, recursive, or bulk-style requests.
If details are missing, use clarification_required and include a concise clarification_question.
"""


@dataclass(frozen=True)
class FileCommand:
    """A classified file-management command."""

    intent: str
    params: Mapping[str, str] = field(default_factory=dict)
    requires_confirmation: bool = False
    clarification_question: str | None = None


@dataclass(frozen=True)
class CommandResult:
    """The result of executing a classified command."""

    success: bool
    message: str
    intent: str
    path: Path | None = None
    requires_confirmation: bool = False


class IntentClassifier(Protocol):
    """Classifies a natural-language question into a structured file command."""

    def classify(self, question: str) -> FileCommand:
        """Return a structured file command."""


class ClassifierUnavailable(RuntimeError):
    """Raised when LLM classification cannot run."""


class LLMFileCommand(BaseModel):
    """Structured output schema returned by the LLM."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal[
        "create_directory",
        "create_file",
        "list_directory",
        "rename_path",
        "copy_path",
        "move_path",
        "delete_path",
        "read_file",
        "write_file",
        "unknown",
        "clarification_required",
    ]
    target: str | None = None
    source: str | None = None
    destination: str | None = None
    content: str | None = None
    requires_confirmation: bool = False
    clarification_question: str | None = None

    def to_file_command(self) -> FileCommand:
        params = {
            key: value
            for key, value in {
                "target": self.target,
                "source": self.source,
                "destination": self.destination,
                "content": self.content,
            }.items()
            if value is not None
        }
        return FileCommand(
            intent=self.intent,
            params=params,
            requires_confirmation=self.requires_confirmation,
            clarification_question=self.clarification_question,
        )


class LLMIntentClassifier:
    """OpenAI Structured Outputs classifier for file-management requests."""

    def __init__(self, client: object | None = None, model: str | None = None) -> None:
        self.client = client
        self.model = model or os.environ.get("FILEMANAGER_MODEL", DEFAULT_MODEL)

    def classify(self, question: str) -> FileCommand:
        client = self._client()

        try:
            response = client.responses.parse(
                model=self.model,
                instructions=CLASSIFIER_INSTRUCTIONS,
                input=[{"role": "user", "content": question}],
                text_format=LLMFileCommand,
            )
        except Exception as exc:  # pragma: no cover - exact SDK exceptions vary by version.
            raise ClassifierUnavailable(f"LLM classification failed: {exc}") from exc

        parsed = getattr(response, "output_parsed", None)
        if not isinstance(parsed, LLMFileCommand):
            raise ClassifierUnavailable("LLM classification returned invalid structured output.")

        return parsed.to_file_command()

    def _client(self) -> object:
        if self.client is not None:
            return self.client

        if not os.environ.get("OPENAI_API_KEY"):
            raise ClassifierUnavailable("OPENAI_API_KEY is required for LLM classification.")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency should be installed by uv.
            raise ClassifierUnavailable("The openai package is required for LLM classification.") from exc

        self.client = OpenAI()
        return self.client


class FileManagerAgent:
    """Classifies user questions and executes safe file-management commands."""

    def __init__(
        self,
        root: Path | str | None = None,
        classifier: IntentClassifier | None = None,
        allow_risky: bool = False,
    ) -> None:
        self.root = Path(root or Path.cwd()).resolve()
        self.classifier = classifier or LLMIntentClassifier()
        self.allow_risky = allow_risky
        self.actions: dict[str, Callable[[FileCommand], CommandResult]] = {
            "create_directory": self._create_directory,
            "create_file": self._create_file,
            "list_directory": self._list_directory,
            "rename_path": self._rename_path,
            "copy_path": self._copy_path,
            "move_path": self._move_path,
            "delete_path": self._delete_path,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "unknown": self._unknown,
            "clarification_required": self._clarification_required,
        }

    def run(self, question: str) -> CommandResult:
        try:
            command = self.classifier.classify(question)
        except ClassifierUnavailable as exc:
            return CommandResult(
                False,
                f"{exc} Please clarify the file operation you want me to perform.",
                "clarification_required",
            )
        return self.execute(command)

    def execute(self, command: FileCommand) -> CommandResult:
        action = self.actions.get(command.intent)
        if action:
            return action(command)

        return CommandResult(
            False,
            "I could not classify that request. Please clarify the file operation you want.",
            "unknown",
        )

    def _create_directory(self, command: FileCommand) -> CommandResult:
        target = self._safe_target(command.params.get("target"))
        self._ensure_can_write_new(target)
        confirmation = self._confirmation_required(command, f"Creating {target.name} requires --yes.")
        if confirmation:
            return confirmation

        target.mkdir(parents=False, exist_ok=False)
        return CommandResult(True, f"Created directory: {target.name}", command.intent, target)

    def _create_file(self, command: FileCommand) -> CommandResult:
        target = self._safe_target(command.params.get("target"))
        self._ensure_can_write_new(target)
        confirmation = self._confirmation_required(command, f"Creating {target.name} requires --yes.")
        if confirmation:
            return confirmation

        target.touch(exist_ok=False)
        return CommandResult(True, f"Created file: {target.name}", command.intent, target)

    def _list_directory(self, command: FileCommand) -> CommandResult:
        target = self._safe_target(command.params.get("target"), allow_root=True) if command.params.get("target") else self.root
        if not target.exists():
            raise ValueError(f"Directory does not exist: {target.name}")
        if not target.is_dir():
            raise ValueError(f"Path is not a directory: {target.name}")

        entries = sorted(path.name for path in target.iterdir())
        message = "\n".join(entries) if entries else "Directory is empty."
        return CommandResult(True, message, command.intent, target)

    def _rename_path(self, command: FileCommand) -> CommandResult:
        source = self._existing_target(command.params.get("source"))
        destination = self._safe_target(command.params.get("destination"))
        self._ensure_can_write_new(destination)
        confirmation = self._confirmation_required(command, f"Renaming {source.name} requires --yes.")
        if confirmation:
            return confirmation

        source.rename(destination)
        return CommandResult(
            True,
            f"Renamed {source.name} to {destination.name}",
            command.intent,
            destination,
        )

    def _copy_path(self, command: FileCommand) -> CommandResult:
        source = self._existing_target(command.params.get("source"))
        destination = self._safe_target(command.params.get("destination"))
        self._ensure_can_write_new(destination)
        confirmation = self._confirmation_required(
            command,
            f"Copying {source.name} requires --yes.",
            force=source.is_dir(),
        )
        if confirmation:
            return confirmation

        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)

        return CommandResult(True, f"Copied {source.name} to {destination.name}", command.intent, destination)

    def _move_path(self, command: FileCommand) -> CommandResult:
        source = self._existing_target(command.params.get("source"))
        destination = self._safe_target(command.params.get("destination"))
        self._ensure_can_write_new(destination)
        confirmation = self._confirmation_required(
            command,
            f"Moving {source.name} requires --yes.",
            force=source.is_dir(),
        )
        if confirmation:
            return confirmation

        shutil.move(str(source), str(destination))
        return CommandResult(True, f"Moved {source.name} to {destination.name}", command.intent, destination)

    def _delete_path(self, command: FileCommand) -> CommandResult:
        target = self._existing_target(command.params.get("target"))
        confirmation = self._confirmation_required(command, f"Deleting {target.name} requires --yes.")
        if confirmation:
            return confirmation

        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

        return CommandResult(True, f"Deleted: {target.name}", command.intent, target)

    def _read_file(self, command: FileCommand) -> CommandResult:
        target = self._existing_target(command.params.get("target"))
        if not target.is_file():
            raise ValueError(f"Path is not a file: {target.name}")
        return CommandResult(True, target.read_text(), command.intent, target)

    def _write_file(self, command: FileCommand) -> CommandResult:
        target = self._safe_target(command.params.get("target"))
        content = command.params.get("content")
        if content is None:
            raise ValueError("Please include content to write.")

        confirmation = self._confirmation_required(
            command,
            f"Overwriting {target.name} requires --yes." if target.exists() else f"Writing {target.name} requires --yes.",
            force=target.exists(),
        )
        if confirmation:
            return confirmation

        target.write_text(content)
        return CommandResult(True, f"Wrote file: {target.name}", command.intent, target)

    def _unknown(self, command: FileCommand) -> CommandResult:
        return CommandResult(
            False,
            "I could not classify that request. Please clarify the file operation you want.",
            command.intent,
        )

    def _clarification_required(self, command: FileCommand) -> CommandResult:
        return CommandResult(
            False,
            command.clarification_question or "Please clarify the file operation you want me to perform.",
            command.intent,
        )

    def _safe_target(self, target: str | None, allow_root: bool = False) -> Path:
        if not target:
            if allow_root:
                return self.root
            raise ValueError("Please include a file or folder name.")

        candidate = Path(target).expanduser()
        if candidate.is_absolute():
            raise ValueError("Absolute paths are not allowed.")

        resolved = (self.root / candidate).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("Path must stay inside the current directory.")

        if resolved == self.root and not allow_root:
            raise ValueError("Please provide a file or folder name inside the current directory.")

        return resolved

    def _existing_target(self, target: str | None) -> Path:
        resolved = self._safe_target(target)
        if not resolved.exists():
            raise ValueError(f"Path does not exist: {resolved.name}")
        return resolved

    def _ensure_can_write_new(self, target: Path) -> None:
        if target.exists():
            raise ValueError(f"Destination already exists: {target.name}")

    def _confirmation_required(
        self,
        command: FileCommand,
        message: str,
        force: bool = False,
    ) -> CommandResult | None:
        if force or command.requires_confirmation or command.intent == "delete_path":
            if not self.allow_risky:
                return CommandResult(False, message, command.intent, requires_confirmation=True)
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the filemanager intent classifier.")
    parser.add_argument("--root", default=None, help="Directory where file operations should run.")
    parser.add_argument("--yes", action="store_true", help="Confirm risky operations such as delete or overwrite.")
    parser.add_argument("question", nargs="+", help="Natural-language request to execute.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    question = " ".join(args.question)

    try:
        result = FileManagerAgent(root=args.root, allow_risky=args.yes).run(question)
    except OSError as exc:
        print(f"Command failed: {exc}")
        return 1
    except ValueError as exc:
        print(f"Invalid request: {exc}")
        return 2

    print(result.message)
    if result.requires_confirmation:
        return 3
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
