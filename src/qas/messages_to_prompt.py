from pydantic import BaseModel
from typing import Mapping, Sequence

from llama_index.llms import ChatMessage, MessageRole
from llama_index.llms.llm import MessagesToPromptType

class MessagesToPromptConverter(BaseModel):
  role_message_templates: Mapping[MessageRole, str]
  """
  Template parameters:

  - `message`: the message text
  """

  generation_prompt: str | None = None

  def __call__(self, messages: Sequence[ChatMessage]):
    return self.format_messages(messages)

  def format_messages(self, messages: Sequence[ChatMessage]) -> str:
    formatted_messages = [self.format_message(m) for m in messages]

    if self.generation_prompt is not None:
      formatted_messages.append(self.generation_prompt)

    return "\n".join(formatted_messages)

  def format_message(self, message: ChatMessage) -> str:
    role_message_template = self.role_message_templates[message.role]
    assert role_message_template is not None
    return role_message_template.format(message=message.content)

def make_mistral_messages_to_prompt_converter() -> MessagesToPromptType:
  return MessagesToPromptConverter(
    role_message_templates={
      MessageRole.SYSTEM: "{message}",
      MessageRole.USER: "[INST]{message}[/INST]",
      MessageRole.ASSISTANT: "{message}",
    }
  )
