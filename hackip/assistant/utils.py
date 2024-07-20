import time
import requests

from openai import OpenAI
from rich.status import Status

from langchain_huggingface import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


class SLMAssistantAPI:
    """
    Integrate Small Language Model from HF / Local.
    Reference: https://colab.research.google.com/drive/1h2505J5H4Y9vngzPD08ppf1ga8sWxLvZ?usp=sharing#scrollTo=uWIc38V4t5gA
    """

    default_prompt = None  # Class variable to store the default prompt
    default_model_id = (
        "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"  # Default model ID for all instances
    )

    def __init__(self) -> None:
        self._prompt = None  # Instance-specific prompt
        self._model_id = None  # Instance-specific model ID

    @property
    def prompt(self):
        return (
            self._prompt if self._prompt is not None else SLMAssistantAPI.default_prompt
        )

    @prompt.setter
    def prompt(self, value):
        if isinstance(value, PromptTemplate):
            self._prompt = value
        else:
            raise ValueError("Prompt must be of type PromptTemplate.")

    @property
    def model_id(self):
        return (
            self._model_id
            if self._model_id is not None
            else SLMAssistantAPI.default_model_id
        )

    @model_id.setter
    def model_id(self, value):
        if isinstance(value, str):
            self._model_id = value
        else:
            raise ValueError("Model ID must be a string.")

    @classmethod
    def add_template(cls, template: str) -> None:
        cls.default_prompt = PromptTemplate(
            template=template, input_variables=["question"]
        )

    def execute(self, question: str) -> str:
        if not self.prompt:
            raise ValueError("No prompt template has been set.")
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id)

        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_length=100
        )

        local_llm = HuggingFacePipeline(pipeline=pipe)

        llm_chain = LLMChain(
            prompt=self.prompt,
            llm=local_llm,
        )
        return llm_chain.invoke(question)


class AssistantAPI(object):
    BASE_URL = "https://api.openai.com/v1/assistants"

    def __init__(self) -> None:
        self._api_key = None
        self._client = None

    @property
    def get_client(self):
        return self._client

    @get_client.setter
    def api_key(self, api_key):
        if not api_key:
            return ValueError("Invalid OpenAI API Key")
        self._api_key = api_key
        self._client = OpenAI(api_key=api_key)

    def list_assistants(self):
        assistant_object = self._client.beta.assistants.list()
        return assistant_object

    def delete_assistant(self, assistant_id):
        """Delete an assistant by ID."""
        delete_url = f"{self.BASE_URL}/{assistant_id}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "assistants=v1",
        }
        response = requests.delete(delete_url, headers=headers)
        if response.status_code == 200:
            print(f"Deleted assistant with ID: {assistant_id}")
        else:
            print(
                f"Failed to delete assistant with ID: {assistant_id}. Status Code: {response.status_code}"
            )

    def delete_all_assistants(self):
        """Delete all assistants."""
        a_list = self.list_assistants()
        assitant_obj_list = a_list.data
        for i in range(len(assitant_obj_list)):
            self.delete_assistant(assitant_obj_list[i].id)

    def select_assistant(self, assistant_id):
        # Use the 'beta.assistants' attribute, not 'Assistant'
        assistant = self._client.beta.assistants.retrieve(assistant_id)
        return assistant.id

    def create_assistant(
        self,
        name: str,
        instructions: str,
        tools,
        model: str = "gpt-4-1106-preview",
        file_ids: list = [],
    ) -> str:
        assistant = self._client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=tools,
            model=model,
            file_ids=file_ids,
        )
        return assistant.id  # Return the assistant ID

    def upload_file(self, filename):
        # Upload a file with an "assistants" purpose
        file_to_upload = self._client.files.create(
            file=open(filename, "rb"), purpose="assistants"
        )
        return file_to_upload

    def get_assistant_by_id(self, assistant_id):
        assistant = self._client.beta.assistants.retrieve(assistant_id)
        return assistant.id

    def get_assistant_by_name(self, assistant_name: str) -> bool:
        a_list = self.list_assistants()
        assitant_obj_list = a_list.data
        for assistant in assitant_obj_list:
            if assistant.name == assistant_name:
                return assistant.id
        return None

    def select_assistant(self, assistant_id):
        return self.get_assistant_by_id(assistant_id)

    def create_thread(self):
        thread = self._client.beta.threads.create()
        return thread

    def create_message(self, thread, content: str, role: str = "user"):
        message = self._client.beta.threads.messages.create(
            thread_id=thread.id,
            role=role,
            content=content,
        )
        return message

    def create_run(self, thread_id, assistant_id):
        run = self._client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
        return run

    def wait_on_run(self, run, thread):
        with Status("[bold green]Running...", spinner="dots"):
            while run.status == "queued" or run.status == "in_progress":
                run = self._client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id,
                )
                time.sleep(1)
        return run

    def get_thread_list(self, thread_id):
        thread_messages = self._client.beta.threads.messages.list(thread_id)
        return thread_messages


if __name__ == "__main__":
    import os

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    obj = SLMAssistantAPI()
    obj.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    obj.add_template("""Question: {question} \n\n Answer: Let's think step by step.""")
    print(obj.execute("What is the capital of France?"))
