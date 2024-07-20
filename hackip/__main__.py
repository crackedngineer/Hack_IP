import re
import argparse
import logging
import platform
import sys

from art import text2art
from .chat import generate_response
from .constants import (
    CREDENTIALS_KEYS,
    GENERATED_REPORT_FOLDER_NAME,
)
from .helpers import create_folder
from .logger_config import setup_logging
from .platforms.os import LinuxOS, MacOS, WindowsOS
from rich import print as rprint
from .assistant.utils import SLMAssistantAPI

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


class HackIP:
    BANNER_TEXT = "HackIP"

    def __init__(self, configuration, advanced_scanning=False):
        self.configuration = configuration
        self.advanced_scanning = advanced_scanning
        SLMAssistantAPI().download_slm()

    def introduction(self):
        try:
            ascii_art = text2art(HackIP.BANNER_TEXT)
            rprint(f"[green]{ascii_art}")
        except Exception as e:
            print(f"Error in generating ASCII art: {e}")

    def get_os_utility(self):
        os_name = platform.system().lower()
        os_util_class = {
            "windows": WindowsOS,
            "linux": LinuxOS,
            "darwin": MacOS,
        }.get(os_name)

        if os_util_class is None:
            rprint("[red]Unsupported Operating System[/red]")
            sys.exit(1)

        cuttly_api_key = self.configuration.get(
            CREDENTIALS_KEYS.CUTTLY_API_KEY.value[0], None
        )

        if cuttly_api_key is None:
            logger.warning("Cuttly API key not found in configuration.")

        return os_util_class(
            cuttly_api_key=cuttly_api_key, advanced_scanning=self.advanced_scanning
        )

    def chat(self):
        openai_key = self.configuration.get(
            CREDENTIALS_KEYS.OPENAI_API_KEY.value[0], None
        )

        if openai_key is None:
            rprint("[red]Error finding OpenAI API key[/red]")
            return

        while True:
            try:
                message = input("Hacker :- ").strip()
                if message.lower() in ["quit", "exit"]:
                    print("Exiting chat.")
                    break

                assistant_message = generate_response(openai_key, message)
                rprint(f"Assistant :- {assistant_message}")

            except KeyboardInterrupt:
                print("\nChat interrupted by user. Exiting...")
                break
            except Exception as e:
                rprint(f"[red]Error: {e}[/red]")

    def start(self, is_fetch=True, is_chat=True):
        self.introduction()

        if is_fetch:
            create_folder(GENERATED_REPORT_FOLDER_NAME)

            os_object = self.get_os_utility()
            os_object.stdout()

        if is_chat:
            self.chat()


class OpenAIKeyAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # Regex for OpenAI key. This is just an example; adjust it as needed.
        pattern = r"^sk-\w+$"
        if not re.match(pattern, values):
            parser.error(f"{values} is not a valid OpenAI API key")
        setattr(namespace, self.dest, values)


def execute():
    try:
        parser = argparse.ArgumentParser(
            description="This is a tool to get IP and system information of a specific device"
        )

        subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

        parser_run = subparsers.add_parser("run", help="Start Execution")
        parser_run.add_argument(
            "-d",
            "--details",
            action="store_true",
            help="Advanced Nmap Scanning of open ports",
        )
        parser_run.add_argument(
            "--chat", action="store_true", help="Chat with the Information"
        )
        parser_run.add_argument(
            "--no_fetch",
            action="store_false",
            help="Chat without fetching system Information",
        )
        parser_run.add_argument(
            "--cuttly_api_key", type=str, help="Cuttly URL shortener API key"
        )
        parser_run.add_argument(
            "--openai_key", type=str, help="OpenAI API Key", action=OpenAIKeyAction
        )

        args = parser.parse_args()

        if args.command == "run":
            rprint("[green]Start Execution...[/green]\n")

            configuration = {
                CREDENTIALS_KEYS.OPENAI_API_KEY.value[0]: args.openai_key,
                CREDENTIALS_KEYS.CUTTLY_API_KEY.value[0]: args.cuttly_api_key,
            }

            HackIP(configuration, args.details).start(
                is_fetch=args.no_fetch, is_chat=args.chat
            )

    except Exception as e:
        print(f"Error during execution: {e}")
