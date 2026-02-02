import asyncio
import contextvars
import copy
import functools
import inspect
import re
import warnings
from collections import defaultdict
from typing import Any, Callable, Coroutine, Literal, Type, Protocol, TypeVar

from pydantic import BaseModel
from termcolor import colored

from train_methods.legacy_autogen.chat import ChatResult, a_initiate_chats, initiate_chats, _post_process_carryover_item, consolidate_chat_info
from train_methods.legacy_autogen.client import ModelClient, OpenAIWrapper
from train_methods.legacy_autogen.stream import IOStream
from train_methods.legacy_autogen.utils import (
    content_str,
    load_basemodels_if_needed, 
    serialize_to_str
)

__all__ = ("ConversableAgent",)

F = TypeVar("F", bound=Callable[..., Any])

def model_dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump()

class SenderRequired(Exception):
    """Exception raised when the sender is required but not provided."""

    def __init__(self, message: str = "Sender is required but not provided."):
        self.message = message
        super().__init__(self.message)

class InvalidCarryOverType(Exception):
    """Exception raised when the carryover type is invalid."""

    def __init__(
        self, message: str = "Carryover should be a string or a list of strings. Not adding carryover to the message."
    ):
        self.message = message
        super().__init__(self.message)


class Agent(Protocol):
    """(In preview) A protocol for Agent.

    An agent can communicate with other agents and perform actions.
    Different agents can differ in what actions they perform in the `receive` method.
    """

    @property
    def name(self) -> str:
        """The name of the agent."""
        ...

    @property
    def description(self) -> str:
        """The description of the agent. Used for the agent's introduction in a group chat setting."""
        ...

    def send(
        self,
        message: dict[str, Any] | str,
        recipient: "Agent",
        request_reply: bool | None = None,
    ) -> None:
        """Send a message to another agent.

        Args:
            message (dict or str): the message to send. If a dict, it should be
            a JSON-serializable and follows the OpenAI's ChatCompletion schema.
            recipient (Agent): the recipient of the message.
            request_reply (bool): whether to request a reply from the recipient.
        """
        ...

    async def a_send(
        self,
        message: dict[str, Any] | str,
        recipient: "Agent",
        request_reply: bool | None = None,
    ) -> None:
        """(Async) Send a message to another agent.

        Args:
            message (dict or str): the message to send. If a dict, it should be
            a JSON-serializable and follows the OpenAI's ChatCompletion schema.
            recipient (Agent): the recipient of the message.
            request_reply (bool): whether to request a reply from the recipient.
        """
        ...

    def receive(
        self,
        message: dict[str, Any] | str,
        sender: "Agent",
        request_reply: bool | None = None,
    ) -> None:
        """Receive a message from another agent.

        Args:
            message (dict or str): the message received. If a dict, it should be
            a JSON-serializable and follows the OpenAI's ChatCompletion schema.
            sender (Agent): the sender of the message.
            request_reply (bool): whether the sender requests a reply.
        """

    async def a_receive(
        self,
        message: dict[str, Any] | str,
        sender: "Agent",
        request_reply: bool | None = None,
    ) -> None:
        """(Async) Receive a message from another agent.

        Args:
            message (dict or str): the message received. If a dict, it should be
            a JSON-serializable and follows the OpenAI's ChatCompletion schema.
            sender (Agent): the sender of the message.
            request_reply (bool): whether the sender requests a reply.
        """
        ...

    def generate_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: Literal["Agent"] | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any] | None:
        """Generate a reply based on the received messages.

        Args:
            messages (list[dict]): a list of messages received from other agents.
                The messages are dictionaries that are JSON-serializable and
                follows the OpenAI's ChatCompletion schema.
            sender: sender of an Agent instance.

        Returns:
            str or dict or None: the generated reply. If None, no reply is generated.
        """

    async def a_generate_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: Literal["Agent"] | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any] | None:
        """(Async) Generate a reply based on the received messages.

        Args:
            messages (list[dict]): a list of messages received from other agents.
                The messages are dictionaries that are JSON-serializable and
                follows the OpenAI's ChatCompletion schema.
            sender: sender of an Agent instance.

        Returns:
            str or dict or None: the generated reply. If None, no reply is generated.
        """

class LLMAgent(Agent, Protocol):
    """(In preview) A protocol for an LLM agent."""

    @property
    def system_message(self) -> str:
        """The system message of this agent."""


class ConversableAgent(LLMAgent):
    """A class for generic conversable agents which can be configured as assistant or user proxy.

    After receiving each message, the agent will send a reply to the sender unless the msg is a termination msg.
    For example, AssistantAgent and UserProxyAgent are subclasses of this class,
    configured with different default settings.
    """

    DEFAULT_CONFIG = False  # False or dict, the default config for llm inference
    MAX_CONSECUTIVE_AUTO_REPLY = 100  # maximum number of consecutive auto replies (subject to future change)

    DEFAULT_SUMMARY_PROMPT = "Summarize the takeaway from the conversation. Do not add any introductory phrases."
    DEFAULT_SUMMARY_METHOD = "last_msg"
    llm_config: dict | Literal[False]

    def __init__(
        self,
        name: str,
        system_message: str | list | None = "You are a helpful AI Assistant.",
        is_termination_msg: Callable[[dict], bool] | None = None,
        llm_config: dict | Literal[False] | None = None,
    ):
        """
        Args:
            name (str): name of the agent.
            system_message (str or list): system message for the ChatCompletion inference.
            is_termination_msg (function): a function that takes a message in the form of a dictionary
                and returns a boolean value indicating if this received message is a termination message.
                The dict can contain the following keys: "content", "role", "name", "function_call".
            max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
                default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
                When set to 0, no auto reply will be generated.
            llm_config (dict or False or None): llm inference configuration.
                Please refer to [OpenAIWrapper.create](/docs/reference/oai/client#create)
                for available options.
                When using OpenAI or Azure OpenAI endpoints, please specify a non-empty 'model' either in `llm_config` or in each config of 'config_list' in `llm_config`.
                To disable llm-based auto reply, set to False.
                When set to None, will use self.DEFAULT_CONFIG, which defaults to False.
        """

        self._name = name
        self._oai_messages = defaultdict(list)

        self._oai_system_message = [{"content": system_message, "role": "system"}]
        self._description = system_message
        self._is_termination_msg = (
            is_termination_msg
            if is_termination_msg is not None
            else (lambda x: content_str(x.get("content")) == "TERMINATE")
        )
        # Take a copy to avoid modifying the given dict
        if isinstance(llm_config, dict):
            llm_config = copy.deepcopy(llm_config)

        self._validate_llm_config(llm_config)

        self.client_cache = None
        self._max_consecutive_auto_reply = self.MAX_CONSECUTIVE_AUTO_REPLY
        self._consecutive_auto_reply_counter = defaultdict(int)
        self._max_consecutive_auto_reply_dict = defaultdict(self.max_consecutive_auto_reply)
        self._reply_func_list = []
        self._human_input = []
        self.reply_at_receive = defaultdict(bool)
        self.register_reply([Agent, None], ConversableAgent.generate_oai_reply)
        self.register_reply([Agent, None], ConversableAgent.a_generate_oai_reply, ignore_async_in_sync_chat=True)

        self.register_reply([Agent, None], ConversableAgent.generate_tool_calls_reply)
        self.register_reply([Agent, None], ConversableAgent.a_generate_tool_calls_reply, ignore_async_in_sync_chat=True)
        self.register_reply([Agent, None], ConversableAgent.generate_function_call_reply)
        self.register_reply(
            [Agent, None], ConversableAgent.a_generate_function_call_reply, ignore_async_in_sync_chat=True
        )
        self.register_reply([Agent, None], ConversableAgent.check_termination_and_human_reply)
        self.register_reply(
            [Agent, None], ConversableAgent.a_check_termination_and_human_reply, ignore_async_in_sync_chat=True
        )

        # Registered hooks are kept in lists, indexed by hookable method, to be called in their order of registration.
        # New hookable methods should be added to this list as required to support new agent capabilities.
        self.hook_lists: dict[str, list[Callable | Callable[..., Coroutine]]] = {
            "process_last_received_message": [],
            "a_process_last_received_message": [],
            "process_all_messages_before_reply": [],
            "a_process_all_messages_before_reply": [],
            "process_message_before_send": [],
            "a_process_message_before_send": [],
        }

    def _validate_llm_config(self, llm_config):
        assert llm_config in (None, False) or isinstance(
            llm_config, dict
        ), "llm_config must be a dict or False or None."
        if llm_config is None:
            llm_config = self.DEFAULT_CONFIG
        self.llm_config = self.DEFAULT_CONFIG if llm_config is None else llm_config
        # TODO: more complete validity check
        if self.llm_config in [{}, {"config_list": []}, {"config_list": [{"model": ""}]}]:
            raise ValueError(
                "When using OpenAI or Azure OpenAI endpoints, specify a non-empty 'model' either in 'llm_config' or in each config of 'config_list'."
            )
        self.client = None if self.llm_config is False else OpenAIWrapper(**self.llm_config)

    @property
    def name(self) -> str:
        """Get the name of the agent."""
        return self._name

    @property
    def description(self) -> str:
        """Get the description of the agent."""
        return self._description

    @description.setter
    def description(self, description: str):
        """Set the description of the agent."""
        self._description = description

    def register_reply(
        self,
        trigger: Type[Agent] | str | Agent | Callable[[Agent], bool] | list,
        reply_func: Callable,
        position: int = 0,
        config: Any | None = None,
        reset_config: Callable | None = None,
        *,
        ignore_async_in_sync_chat: bool = False,
        remove_other_reply_funcs: bool = False,
    ):
        """Register a reply function.

        The reply function will be called when the trigger matches the sender.
        The function registered later will be checked earlier by default.
        To change the order, set the position to a positive integer.

        Both sync and async reply functions can be registered. The sync reply function will be triggered
        from both sync and async chats. However, an async reply function will only be triggered from async
        chats (initiated with `ConversableAgent.a_initiate_chat`). If an `async` reply function is registered
        and a chat is initialized with a sync function, `ignore_async_in_sync_chat` determines the behaviour as follows:
                if `ignore_async_in_sync_chat` is set to `False` (default value), an exception will be raised, and
                if `ignore_async_in_sync_chat` is set to `True`, the reply function will be ignored.

        Args:
            trigger (Agent class, str, Agent instance, callable, or list): the trigger.
                    If a class is provided, the reply function will be called when the sender is an instance of the class.
                    If a string is provided, the reply function will be called when the sender's name matches the string.
                    If an agent instance is provided, the reply function will be called when the sender is the agent instance.
                    If a callable is provided, the reply function will be called when the callable returns True.
                    If a list is provided, the reply function will be called when any of the triggers in the list is activated.
                    If None is provided, the reply function will be called only when the sender is None.
                    Note: Be sure to register `None` as a trigger if you would like to trigger an auto-reply function with non-empty messages and `sender=None`.
            reply_func (Callable): the reply function.
                The function takes a recipient agent, a list of messages, a sender agent and a config as input and returns a reply message.
            position (int): the position of the reply function in the reply function list.
                The function registered later will be checked earlier by default.
                To change the order, set the position to a positive integer.
            config (Any): the config to be passed to the reply function.
                When an agent is reset, the config will be reset to the original value.
            reset_config (Callable): the function to reset the config.
                The function returns None. Signature: ```def reset_config(config: Any)```
            ignore_async_in_sync_chat (bool): whether to ignore the async reply function in sync chats. If `False`, an exception
                will be raised if an async reply function is registered and a chat is initialized with a sync
                function.
            remove_other_reply_funcs (bool): whether to remove other reply functions when registering this reply function.
        """
        if not isinstance(trigger, (type, str, Agent, Callable, list)):
            raise ValueError("trigger must be a class, a string, an agent, a callable or a list.")
        if remove_other_reply_funcs:
            self._reply_func_list.clear()
        self._reply_func_list.insert(
            position,
            {
                "trigger": trigger,
                "reply_func": reply_func,
                "config": copy.copy(config),
                "init_config": config,
                "reset_config": reset_config,
                "ignore_async_in_sync_chat": ignore_async_in_sync_chat and inspect.iscoroutinefunction(reply_func),
            },
        )


    @property
    def system_message(self) -> str:
        """Return the system message."""
        return self._oai_system_message[0]["content"]

    def max_consecutive_auto_reply(self, sender: Agent | None = None) -> int:
        """The maximum number of consecutive auto replies."""
        return self._max_consecutive_auto_reply if sender is None else self._max_consecutive_auto_reply_dict[sender]

    @property
    def chat_messages(self) -> dict[Agent, list[dict]]:
        """A dictionary of conversations from agent to list of messages."""
        return self._oai_messages

    def last_message(self, agent: Agent | None = None) -> dict | None:
        """The last message exchanged with the agent.

        Args:
            agent (Agent): The agent in the conversation.
                If None and more than one agent's conversations are found, an error will be raised.
                If None and only one conversation is found, the last message of the only conversation will be returned.

        Returns:
            The last message exchanged with the agent.
        """
        if agent is None:
            n_conversations = len(self._oai_messages)
            if n_conversations == 0:
                return None
            if n_conversations == 1:
                for conversation in self._oai_messages.values():
                    return conversation[-1]
            raise ValueError("More than one conversation is found. Please specify the sender to get the last message.")
        if agent not in self._oai_messages.keys():
            raise KeyError(
                f"The agent '{agent.name}' is not present in any conversation. No history available for this agent."
            )
        return self._oai_messages[agent][-1]

    @staticmethod
    def _message_to_dict(message: dict | str) -> dict:
        """Convert a message to a dictionary.

        The message can be a string or a dictionary. The string will be put in the "content" field of the new dictionary.
        """
        if isinstance(message, str):
            return {"content": message}
        elif isinstance(message, dict):
            return message
        else:
            return dict(message)

    @staticmethod
    def _normalize_name(name):
        """
        LLMs sometimes ask functions while ignoring their own format requirements, this function should be used to replace invalid characters with "_".

        Prefer _assert_valid_name for validating user configuration or input
        """
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64]

    @staticmethod
    def _assert_valid_name(name):
        """
        Ensure that configured names are valid, raises ValueError if not.

        For munging LLM responses use _normalize_name to ensure LLM specified names don't break the API.
        """
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise ValueError(f"Invalid name: {name}. Only letters, numbers, '_' and '-' are allowed.")
        if len(name) > 64:
            raise ValueError(f"Invalid name: {name}. Name must be less than 64 characters.")
        return name

    def _append_oai_message(self, message: dict | str, role, conversation_id: Agent, is_sending: bool) -> bool:
        """Append a message to the ChatCompletion conversation.

        If the message received is a string, it will be put in the "content" field of the new dictionary.
        If the message received is a dictionary but does not have any of the three fields "content", "function_call", or "tool_calls",
            this message is not a valid ChatCompletion message.
        If only "function_call" or "tool_calls" is provided, "content" will be set to None if not provided, and the role of the message will be forced "assistant".

        Args:
            message (dict or str): message to be appended to the ChatCompletion conversation.
            role (str): role of the message, can be "assistant" or "function".
            conversation_id (Agent): id of the conversation, should be the recipient or sender.
            is_sending (bool): If the agent (aka self) is sending to the conversation_id agent, otherwise receiving.

        Returns:
            bool: whether the message is appended to the ChatCompletion conversation.
        """
        message = self._message_to_dict(message)
        # create oai message to be appended to the oai conversation that can be passed to oai directly.
        oai_message = {
            k: message[k]
            for k in ("content", "function_call", "tool_calls", "tool_responses", "tool_call_id", "name", "context")
            if k in message and message[k] is not None
        }
        if "content" not in oai_message:
            if "function_call" in oai_message or "tool_calls" in oai_message:
                oai_message["content"] = None  # if only function_call is provided, content will be set to None.
            else:
                return False

        if message.get("role") in ["function", "tool"]:
            oai_message["role"] = message.get("role")
        elif "override_role" in message:
            # If we have a direction to override the role then set the
            # role accordingly. Used to customise the role for the
            # select speaker prompt.
            oai_message["role"] = message.get("override_role")
        else:
            oai_message["role"] = role

        if oai_message.get("function_call", False) or oai_message.get("tool_calls", False):
            oai_message["role"] = "assistant"  # only messages with role 'assistant' can have a function call.
        elif "name" not in oai_message:
            # If we don't have a name field, append it
            if is_sending:
                oai_message["name"] = self.name
            else:
                oai_message["name"] = conversation_id.name

        self._oai_messages[conversation_id].append(oai_message)

        return True

    def _process_message_before_send(
        self, message: dict | str, recipient: Agent
    ) -> dict | str:
        """Process the message before sending it to the recipient."""
        hook_list = self.hook_lists["process_message_before_send"]
        for hook in hook_list:
            if inspect.iscoroutinefunction(hook):
                continue
            message = hook(
                sender=self, message=message, recipient=recipient, silent=False
            )
        return message

    async def _a_process_message_before_send(
        self, message: dict | str, recipient: Agent
    ) -> dict | str:
        """(async) Process the message before sending it to the recipient."""
        hook_list = self.hook_lists["a_process_message_before_send"]
        for hook in hook_list:
            if not inspect.iscoroutinefunction(hook):
                continue
            message = await hook(sender=self, message=message, recipient=recipient, silent=False)
        return message

    def send(
        self,
        message: dict | str,
        recipient: Agent,
        request_reply: bool | None = None,
        silent: bool | None = False,
    ):
        """Send a message to another agent.

        Args:
            message (dict or str): message to be sent.
                The message could contain the following fields:
                - content (str or List): Required, the content of the message. (Can be None)
                - function_call (str): the name of the function to be called.
                - name (str): the name of the function to be called.
                - role (str): the role of the message, any role that is not "function"
                    will be modified to "assistant".
                - context (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](../oai/client#create).
                    Next time, one agent can send a message B with a different "use_tool_msg".
                    Then the content of message A will be refreshed to the new "use_tool_msg".
                    So effectively, this provides a way for an agent to send a "link" and modify
                    the content of the "link" later.
            recipient (Agent): the recipient of the message.
            request_reply (bool or None): whether to request a reply from the recipient.
            silent (bool or None): (Experimental) whether to print the message sent.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        """
        message = self._process_message_before_send(message, recipient)
        # When the agent composes and sends the message, the role of the message is "assistant"
        # unless it's "function".
        valid = self._append_oai_message(message, "assistant", recipient, is_sending=True)
        if valid:
            recipient.receive(message, self, request_reply, silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

    async def a_send(
        self,
        message: dict | str,
        recipient: Agent,
        request_reply: bool | None = None,
        silent: bool | None = False,
    ):
        """(async) Send a message to another agent."""
        message = await self._a_process_message_before_send(message, recipient)
        # When the agent composes and sends the message, the role of the message is "assistant"
        # unless it's "function".
        valid = self._append_oai_message(message, "assistant", recipient, is_sending=True)
        if valid:
            await recipient.a_receive(message, self, request_reply, silent)
        else:
            raise ValueError(
                "Message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

    def _print_received_message(self, message: dict | str, sender: Agent):
        iostream = IOStream.get_default()
        # print the message received
        iostream.print(colored(sender.name, "yellow"), "(to", f"{self.name}):\n", flush=True)
        message = self._message_to_dict(message)

        if message.get("tool_responses"):  # Handle tool multi-call responses
            for tool_response in message["tool_responses"]:
                self._print_received_message(tool_response, sender)
            if message.get("role") == "tool":
                return  # If role is tool, then content is just a concatenation of all tool_responses

        if message.get("role") in ["function", "tool"]:
            if message["role"] == "function":
                id_key = "name"
            else:
                id_key = "tool_call_id"
            id = message.get(id_key, "No id found")
            func_print = f"***** Response from calling {message['role']} ({id}) *****"
            iostream.print(colored(func_print, "green"), flush=True)
            iostream.print(message["content"], flush=True)
            iostream.print(colored("*" * len(func_print), "green"), flush=True)
        else:
            content = message.get("content")
            if content is not None:
                if "context" in message:
                    content = OpenAIWrapper.instantiate(
                        content,
                        message["context"],
                        self.llm_config and self.llm_config.get("allow_format_str_template", False),
                    )
                iostream.print(content_str(content), flush=True)
            if "function_call" in message and message["function_call"]:
                function_call = dict(message["function_call"])
                func_print = (
                    f"***** Suggested function call: {function_call.get('name', '(No function name found)')} *****"
                )
                iostream.print(colored(func_print, "green"), flush=True)
                iostream.print(
                    "Arguments: \n",
                    function_call.get("arguments", "(No arguments found)"),
                    flush=True,
                    sep="",
                )
                iostream.print(colored("*" * len(func_print), "green"), flush=True)
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    id = tool_call.get("id", "No tool call id found")
                    function_call = dict(tool_call.get("function", {}))
                    func_print = f"***** Suggested tool call ({id}): {function_call.get('name', '(No function name found)')} *****"
                    iostream.print(colored(func_print, "green"), flush=True)
                    iostream.print(
                        "Arguments: \n",
                        function_call.get("arguments", "(No arguments found)"),
                        flush=True,
                        sep="",
                    )
                    iostream.print(colored("*" * len(func_print), "green"), flush=True)

        iostream.print("\n", "-" * 80, flush=True, sep="")

    def _process_received_message(self, message: dict | str, sender: Agent):
        # When the agent receives a message, the role of the message is "user". (If 'role' exists and is 'function', it will remain unchanged.)
        valid = self._append_oai_message(message, "user", sender, is_sending=False)

        if not valid:
            raise ValueError(
                "Received message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )

        self._print_received_message(message, sender)

    def receive(
        self,
        message: dict | str,
        sender: Agent,
        request_reply: bool | None = None,
        silent: bool | None = False,
    ):
        """Receive a message from another agent.

        Once a message is received, this function sends a reply to the sender or stop.
        The reply can be generated automatically or entered manually by a human.

        Args:
            message (dict or str): message from the sender. If the type is dict, it may contain the following reserved fields (either content or function_call need to be provided).
                1. "content": content of the message, can be None.
                2. "function_call": a dictionary containing the function name and arguments. (deprecated in favor of "tool_calls")
                3. "tool_calls": a list of dictionaries containing the function name and arguments.
                4. "role": role of the message, can be "assistant", "user", "function", "tool".
                    This field is only needed to distinguish between "function" or "assistant"/"user".
                5. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
                6. "context" (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](../oai/client#create).
            sender: sender of an Agent instance.
            request_reply (bool or None): whether a reply is requested from the sender.
                If None, the value is determined by `self.reply_at_receive[sender]`.
            silent (bool or None): (Experimental) whether to print the message received.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        """
        self._process_received_message(message, sender)
        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            return
        reply = self.generate_reply(messages=self.chat_messages[sender], sender=sender)
        if reply is not None:
            self.send(reply, sender, silent=silent)

    async def a_receive(
        self,
        message: dict | str,
        sender: Agent,
        request_reply: bool | None = None,
        silent: bool | None = False,
    ):
        """(async) Receive a message from another agent.

        Once a message is received, this function sends a reply to the sender or stop.
        The reply can be generated automatically or entered manually by a human.

        Args:
            message (dict or str): message from the sender. If the type is dict, it may contain the following reserved fields (either content or function_call need to be provided).
                1. "content": content of the message, can be None.
                2. "function_call": a dictionary containing the function name and arguments. (deprecated in favor of "tool_calls")
                3. "tool_calls": a list of dictionaries containing the function name and arguments.
                4. "role": role of the message, can be "assistant", "user", "function".
                    This field is only needed to distinguish between "function" or "assistant"/"user".
                5. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
                6. "context" (dict): the context of the message, which will be passed to
                    [OpenAIWrapper.create](../oai/client#create).
            sender: sender of an Agent instance.
            request_reply (bool or None): whether a reply is requested from the sender.
                If None, the value is determined by `self.reply_at_receive[sender]`.
            silent (bool or None): (Experimental) whether to print the message received.

        Raises:
            ValueError: if the message can't be converted into a valid ChatCompletion message.
        """
        self._process_received_message(message, sender, silent)
        if request_reply is False or request_reply is None and self.reply_at_receive[sender] is False:
            return
        reply = await self.a_generate_reply(sender=sender)
        if reply is not None:
            await self.a_send(reply, sender, silent=silent)

    def _prepare_chat(
        self,
        recipient: "ConversableAgent",
        clear_history: bool,
        prepare_recipient: bool = True,
        reply_at_receive: bool = True,
    ) -> None:
        self.reset_consecutive_auto_reply_counter(recipient)
        self.reply_at_receive[recipient] = reply_at_receive
        if clear_history:
            self.clear_history(recipient)
            self._human_input = []
        if prepare_recipient:
            recipient._prepare_chat(self, clear_history, False, reply_at_receive)

    def _raise_exception_on_async_reply_functions(self) -> None:
        """Raise an exception if any async reply functions are registered.

        Raises:
            RuntimeError: if any async reply functions are registered.
        """
        reply_functions = {
            f["reply_func"] for f in self._reply_func_list if not f.get("ignore_async_in_sync_chat", False)
        }

        async_reply_functions = [f for f in reply_functions if inspect.iscoroutinefunction(f)]
        if async_reply_functions:
            msg = (
                "Async reply functions can only be used with ConversableAgent.a_initiate_chat(). The following async reply functions are found: "
                + ", ".join([f.__name__ for f in async_reply_functions])
            )

            raise RuntimeError(msg)

    def initiate_chat(
        self,
        recipient: "ConversableAgent",
        clear_history: bool = True,
        silent: bool | None = False,
        max_turns: int | None = None,
        summary_method: str | Callable | None = DEFAULT_SUMMARY_METHOD,
        summary_args: dict | None = {},
        message: dict | str | Callable | None = None,
        **kwargs,
    ) -> ChatResult:
        """Initiate a chat with the recipient agent.

        Reset the consecutive auto reply counter.
        If `clear_history` is True, the chat history with the recipient agent will be cleared.


        Args:
            recipient: the recipient agent.
            clear_history (bool): whether to clear the chat history with the agent. Default is True.
            silent (bool or None): (Experimental) whether to print the messages for this conversation. Default is False.
            max_turns (int or None): the maximum number of turns for the chat between the two agents. One turn means one conversation round trip. Note that this is different from
                [max_consecutive_auto_reply](#max_consecutive_auto_reply) which is the maximum number of consecutive auto replies; and it is also different from [max_rounds in GroupChat](./groupchat#groupchat-objects) which is the maximum number of rounds in a group chat session.
                If max_turns is set to None, the chat will continue until a termination condition is met. Default is None.
            summary_method (str or callable): a method to get a summary from the chat. Default is DEFAULT_SUMMARY_METHOD, i.e., "last_msg".

            Supported strings are "last_msg" and "reflection_with_llm":
                - when set to "last_msg", it returns the last message of the dialog as the summary.
                - when set to "reflection_with_llm", it returns a summary extracted using an llm client.
                    `llm_config` must be set in either the recipient or sender.

            A callable summary_method should take the recipient and sender agent in a chat as input and return a string of summary.
            summary_args (dict): a dictionary of arguments to be passed to the summary_method.
                One example key is "summary_prompt", and value is a string of text used to prompt a LLM-based agent (the sender or receiver agent) to reflect
                on the conversation and extract a summary when summary_method is "reflection_with_llm".
                The default summary_prompt is DEFAULT_SUMMARY_PROMPT, i.e., "Summarize takeaway from the conversation. Do not add any introductory phrases. If the intended request is NOT properly addressed, please point it out."
                Another available key is "summary_role", which is the role of the message sent to the agent in charge of summarizing. Default is "system".
            message (str, dict or Callable): the initial message to be sent to the recipient. Needs to be provided. Otherwise, input() will be called to get the initial message.
                - If a string or a dict is provided, it will be used as the initial message.        `generate_init_message` is called to generate the initial message for the agent based on this string and the context.
                    If dict, it may contain the following reserved fields (either content or tool_calls need to be provided).

                        1. "content": content of the message, can be None.
                        2. "function_call": a dictionary containing the function name and arguments. (deprecated in favor of "tool_calls")
                        3. "tool_calls": a list of dictionaries containing the function name and arguments.
                        4. "role": role of the message, can be "assistant", "user", "function".
                            This field is only needed to distinguish between "function" or "assistant"/"user".
                        5. "name": In most cases, this field is not needed. When the role is "function", this field is needed to indicate the function name.
                        6. "context" (dict): the context of the message, which will be passed to
                            [OpenAIWrapper.create](../oai/client#create).

                - If a callable is provided, it will be called to get the initial message in the form of a string or a dict.
                    If the returned type is dict, it may contain the reserved fields mentioned above.

            **kwargs: any additional information. It has the following reserved fields:
                - "carryover": a string or a list of string to specify the carryover information to be passed to this chat.
                    If provided, we will combine this carryover (by attaching a "context: " string and the carryover content after the message content) with the "message" content when generating the initial chat
                    message in `generate_init_message`.
                - "verbose": a boolean to specify whether to print the message and carryover in a chat. Default is False.

        Raises:
            RuntimeError: if any async reply functions are registered and not ignored in sync chat.

        Returns:
            ChatResult: an ChatResult object.
        """
        _chat_info = locals().copy()
        _chat_info["sender"] = self
        consolidate_chat_info(_chat_info, uniform_sender=self)
        for agent in [self, recipient]:
            agent._raise_exception_on_async_reply_functions()
            agent.previous_cache = agent.client_cache
            agent.client_cache = None
        if isinstance(max_turns, int):
            self._prepare_chat(recipient, clear_history, reply_at_receive=False)
            for _ in range(max_turns):
                if _ == 0:
                    if isinstance(message, Callable):
                        msg2send = message(_chat_info["sender"], _chat_info["recipient"], kwargs)
                    else:
                        msg2send = self.generate_init_message(message, **kwargs)
                else:
                    msg2send = self.generate_reply(messages=self.chat_messages[recipient], sender=recipient)
                if msg2send is None:
                    break
                self.send(msg2send, recipient, request_reply=True, silent=silent)
        else:
            self._prepare_chat(recipient, clear_history)
            if isinstance(message, Callable):
                msg2send = message(_chat_info["sender"], _chat_info["recipient"], kwargs)
            else:
                msg2send = self.generate_init_message(message, **kwargs)
            self.send(msg2send, recipient, silent=silent)
        summary = self._summarize_chat(
            recipient,
        )
        for agent in [self, recipient]:
            agent.client_cache = agent.previous_cache
            agent.previous_cache = None
        chat_result = ChatResult(
            chat_history=self.chat_messages[recipient],
            summary=summary,
            cost=None,
            human_input=self._human_input,
        )
        return chat_result

    async def a_initiate_chat(
        self,
        recipient: "ConversableAgent",
        clear_history: bool = True,
        silent: bool | None = False,
        max_turns: int | None = None,
        summary_method: str | Callable | None = DEFAULT_SUMMARY_METHOD,
        summary_args: dict | None = {},
        message: str | Callable | None = None,
        **kwargs,
    ) -> ChatResult:
        """(async) Initiate a chat with the recipient agent.

        Reset the consecutive auto reply counter.
        If `clear_history` is True, the chat history with the recipient agent will be cleared.
        `a_generate_init_message` is called to generate the initial message for the agent.

        Args: Please refer to `initiate_chat`.

        Returns:
            ChatResult: an ChatResult object.
        """
        _chat_info = locals().copy()
        _chat_info["sender"] = self
        consolidate_chat_info(_chat_info, uniform_sender=self)
        for agent in [self, recipient]:
            agent.previous_cache = agent.client_cache
            agent.client_cache = None
        if isinstance(max_turns, int):
            self._prepare_chat(recipient, clear_history, reply_at_receive=False)
            for _ in range(max_turns):
                if _ == 0:
                    if isinstance(message, Callable):
                        msg2send = message(_chat_info["sender"], _chat_info["recipient"], kwargs)
                    else:
                        msg2send = await self.a_generate_init_message(message, **kwargs)
                else:
                    msg2send = await self.a_generate_reply(messages=self.chat_messages[recipient], sender=recipient)
                if msg2send is None:
                    break
                await self.a_send(msg2send, recipient, request_reply=True, silent=silent)
        else:
            self._prepare_chat(recipient, clear_history)
            if isinstance(message, Callable):
                msg2send = message(_chat_info["sender"], _chat_info["recipient"], kwargs)
            else:
                msg2send = await self.a_generate_init_message(message, **kwargs)
            await self.a_send(msg2send, recipient, silent=silent)
        summary = self._summarize_chat(
            recipient,
        )
        for agent in [self, recipient]:
            agent.client_cache = agent.previous_cache
            agent.previous_cache = None
        chat_result = ChatResult(
            chat_history=self.chat_messages[recipient],
            summary=summary,
            cost=None,
            human_input=self._human_input,
        )
        return chat_result

    def _summarize_chat(
        self,
        recipient: Agent | None = None,
    ) -> str:
        """Get a chat summary from an agent participating in a chat.

        Args:
            summary_method (str or callable): the summary_method to get the summary.
            summary_args (dict): a dictionary of arguments to be passed to the summary_method.
            recipient: the recipient agent in a chat.
            prompt (str): the prompt used to get a summary when summary_method is "reflection_with_llm".

        Returns:
            str: a chat summary from the agent.
        """
        return self._last_msg_as_summary(self, recipient)

    @staticmethod
    def _last_msg_as_summary(sender, recipient) -> str:
        """Get a chat summary from the last message of the recipient."""
        summary = ""
        try:
            content = recipient.last_message(sender)["content"]
            if isinstance(content, str):
                summary = content.replace("TERMINATE", "")
            elif isinstance(content, list):
                # Remove the `TERMINATE` word in the content list.
                summary = "\n".join(
                    x["text"].replace("TERMINATE", "") for x in content if isinstance(x, dict) and "text" in x
                )
        except (IndexError, AttributeError) as e:
            warnings.warn(f"Cannot extract summary using last_msg: {e}. Using an empty str as summary.", UserWarning)
        return summary

    def _reflection_with_llm(
        self,
        prompt,
        messages,
        llm_agent: Agent | None = None,
        role: str | None = None,
    ) -> str:
        """Get a chat summary using reflection with an llm client based on the conversation history.

        Args:
            prompt (str): The prompt (in this method it is used as system prompt) used to get the summary.
            messages (list): The messages generated as part of a chat conversation.
            llm_agent: the agent with an llm client.
            role (str): the role of the message, usually "system" or "user". Default is "system".
        """
        if not role:
            role = "system"

        system_msg = [
            {
                "role": role,
                "content": prompt,
            }
        ]

        messages = messages + system_msg
        if llm_agent and llm_agent.client is not None:
            llm_client = llm_agent.client
        elif self.client is not None:
            llm_client = self.client
        else:
            raise ValueError("No OpenAIWrapper client is found.")
        response = self._generate_oai_reply_from_client(llm_client=llm_client, messages=messages)
        return response

    def _check_chat_queue_for_sender(self, chat_queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Check the chat queue and add the "sender" key if it's missing.

        Args:
            chat_queue (list[dict[str, Any]]): A list of dictionaries containing chat information.

        Returns:
            list[dict[str, Any]]: A new list of dictionaries with the "sender" key added if it was missing.
        """
        chat_queue_with_sender = []
        for chat_info in chat_queue:
            if chat_info.get("sender") is None:
                chat_info["sender"] = self
            chat_queue_with_sender.append(chat_info)
        return chat_queue_with_sender

    def initiate_chats(self, chat_queue: list[dict[str, Any]]) -> list[ChatResult]:
        """(Experimental) Initiate chats with multiple agents.

        Args:
            chat_queue (list[dict]): a list of dictionaries containing the information of the chats.
                Each dictionary should contain the input arguments for [`initiate_chat`](conversable_agent#initiate_chat)

        Returns: a list of ChatResult objects corresponding to the finished chats in the chat_queue.
        """
        _chat_queue = self._check_chat_queue_for_sender(chat_queue)
        self._finished_chats = initiate_chats(_chat_queue)
        return self._finished_chats

    async def a_initiate_chats(self, chat_queue: list[dict[str, Any]]) -> dict[int, ChatResult]:
        _chat_queue = self._check_chat_queue_for_sender(chat_queue)
        self._finished_chats = await a_initiate_chats(_chat_queue)
        return self._finished_chats

    def get_chat_results(self, chat_index: int | None = None) -> list[ChatResult] | ChatResult:
        """A summary from the finished chats of particular agents."""
        if chat_index is not None:
            return self._finished_chats[chat_index]
        else:
            return self._finished_chats

    def reset(self):
        """Reset the agent."""
        self.clear_history()
        self.reset_consecutive_auto_reply_counter()
        self.stop_reply_at_receive()
        if self.client is not None:
            self.client.clear_usage_summary()
        for reply_func_tuple in self._reply_func_list:
            if reply_func_tuple["reset_config"] is not None:
                reply_func_tuple["reset_config"](reply_func_tuple["config"])
            else:
                reply_func_tuple["config"] = copy.copy(reply_func_tuple["init_config"])

    def stop_reply_at_receive(self, sender: Agent | None = None):
        """Reset the reply_at_receive of the sender."""
        if sender is None:
            self.reply_at_receive.clear()
        else:
            self.reply_at_receive[sender] = False

    def reset_consecutive_auto_reply_counter(self, sender: Agent | None = None):
        """Reset the consecutive_auto_reply_counter of the sender."""
        if sender is None:
            self._consecutive_auto_reply_counter.clear()
        else:
            self._consecutive_auto_reply_counter[sender] = 0

    def clear_history(self, recipient: Agent | None = None, nr_messages_to_preserve: int | None = None):
        """Clear the chat history of the agent.

        Args:
            recipient: the agent with whom the chat history to clear. If None, clear the chat history with all agents.
            nr_messages_to_preserve: the number of newest messages to preserve in the chat history.
        """
        iostream = IOStream.get_default()
        if recipient is None:
            if nr_messages_to_preserve:
                for key in self._oai_messages:
                    nr_messages_to_preserve_internal = nr_messages_to_preserve
                    # if breaking history between function call and function response, save function call message
                    # additionally, otherwise openai will return error
                    first_msg_to_save = self._oai_messages[key][-nr_messages_to_preserve_internal]
                    if "tool_responses" in first_msg_to_save:
                        nr_messages_to_preserve_internal += 1
                        iostream.print(
                            f"Preserving one more message for {self.name} to not divide history between tool call and "
                            f"tool response."
                        )
                    # Remove messages from history except last `nr_messages_to_preserve` messages.
                    self._oai_messages[key] = self._oai_messages[key][-nr_messages_to_preserve_internal:]
            else:
                self._oai_messages.clear()
        else:
            self._oai_messages[recipient].clear()
            if nr_messages_to_preserve:
                iostream.print(
                    colored(
                        "WARNING: `nr_preserved_messages` is ignored when clearing chat history with a specific agent.",
                        "yellow",
                    ),
                    flush=True,
                )

    def generate_oai_reply(
        self,
        messages: list[dict] | None = None,
        sender: Agent | None = None,
        config: OpenAIWrapper | None = None,
    ) -> tuple[bool, str | dict | None]:
        """Generate a reply using autogen.oai."""
        client = self.client if config is None else config
        if client is None:
            return False, None
        if messages is None:
            messages = self._oai_messages[sender]
        extracted_response = self._generate_oai_reply_from_client(
            client, self._oai_system_message + messages
        )
        return (False, None) if extracted_response is None else (True, extracted_response)

    def _generate_oai_reply_from_client(self, llm_client, messages) -> str | dict | None:
        all_messages = []
        for message in messages:
            tool_responses = message.get("tool_responses", [])
            if tool_responses:
                all_messages += tool_responses
                # tool role on the parent message means the content is just concatenation of all of the tool_responses
                if message.get("role") != "tool":
                    all_messages.append({key: message[key] for key in message if key != "tool_responses"})
            else:
                all_messages.append(message)

        response = llm_client.create(
            context=messages[-1].pop("context", None), messages=all_messages, agent=self
        )
        extracted_response = llm_client.extract_text_or_completion_object(response)[0]

        if extracted_response is None:
            warnings.warn(f"Extracted_response from {response} is None.", UserWarning)
            return None
        # ensure function and tool calls will be accepted when sent back to the LLM
        if not isinstance(extracted_response, str) and hasattr(extracted_response, "model_dump"):
            extracted_response = model_dump(extracted_response)
        if isinstance(extracted_response, dict):
            if extracted_response.get("function_call"):
                extracted_response["function_call"]["name"] = self._normalize_name(
                    extracted_response["function_call"]["name"]
                )
            for tool_call in extracted_response.get("tool_calls") or []:
                tool_call["function"]["name"] = self._normalize_name(tool_call["function"]["name"])
                # Remove id and type if they are not present.
                # This is to make the tool call object compatible with Mistral API.
                if tool_call.get("id") is None:
                    tool_call.pop("id")
                if tool_call.get("type") is None:
                    tool_call.pop("type")
        return extracted_response

    async def a_generate_oai_reply(
        self,
        messages: list[dict] | None = None,
        sender: Agent | None = None,
        config: Any | None = None,
    ) -> tuple[bool, str | dict | None]:
        """Generate a reply using autogen.oai asynchronously."""
        iostream = IOStream.get_default()
        parent_context = contextvars.copy_context()

        def _generate_oai_reply(
            self, iostream: IOStream, *args: Any, **kwargs: Any
        ) -> tuple[bool, str | dict | None]:
            with IOStream.set_default(iostream):
                return self.generate_oai_reply(*args, **kwargs)

        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: parent_context.run(
                _generate_oai_reply, self=self, iostream=iostream, messages=messages, sender=sender, config=config
            ),
        )

    def generate_function_call_reply(
        self,
        messages: list[dict] | None = None,
        sender: Agent | None = None,
        config: Any | None = None,
    ) -> tuple[bool, dict | None]:
        """
        Generate a reply using function call.

        "function_call" replaced by "tool_calls" as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
        See https://platform.openai.com/docs/api-reference/chat/create#chat-create-functions
        """
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        if "function_call" in message and message["function_call"]:
            func_return = self.execute_function(message["function_call"])
            return True, func_return
        return False, None

    async def a_generate_function_call_reply(
        self,
        messages: list[dict] | None = None,
        sender: Agent | None = None,
        config: Any | None = None,
    ) -> tuple[bool, dict | None]:
        """
        Generate a reply using async function call.

        "function_call" replaced by "tool_calls" as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
        See https://platform.openai.com/docs/api-reference/chat/create#chat-create-functions
        """
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        func_call = message.get("function_call")
        if func_call:
            func_return = self.execute_function(func_call)
            return True, func_return

        return False, None

    def _str_for_tool_response(self, tool_response):
        return str(tool_response.get("content", ""))

    def generate_tool_calls_reply(
        self,
        messages: list[dict] | None = None,
        sender: Agent | None = None,
        config: Any | None = None,
    ) -> tuple[bool, dict | None]:
        """Generate a reply using tool call."""
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        tool_returns = []
        for tool_call in message.get("tool_calls", []):
            function_call = tool_call.get("function", {})
            func_return = self.execute_function(function_call)
            content = func_return.get("content", "")
            if content is None:
                content = ""
            tool_call_id = tool_call.get("id", None)
            if tool_call_id is not None:
                tool_call_response = {
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "content": content,
                }
            else:
                # Do not include tool_call_id if it is not present.
                # This is to make the tool call object compatible with Mistral API.
                tool_call_response = {
                    "role": "tool",
                    "content": content,
                }
            tool_returns.append(tool_call_response)
        if tool_returns:
            return True, {
                "role": "tool",
                "tool_responses": tool_returns,
                "content": "\n\n".join([self._str_for_tool_response(tool_return) for tool_return in tool_returns]),
            }
        return False, None

    async def _a_execute_tool_call(self, tool_call):
        id = tool_call["id"]
        function_call = tool_call.get("function", {})
        func_return = await self.a_execute_function(function_call)
        return {
            "tool_call_id": id,
            "role": "tool",
            "content": func_return.get("content", ""),
        }

    async def a_generate_tool_calls_reply(
        self,
        messages: list[dict] | None = None,
        sender: Agent | None = None,
        config: Any | None = None,
    ) -> tuple[bool, dict | None]:
        """Generate a reply using async function call."""
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        async_tool_calls = []
        for tool_call in message.get("tool_calls", []):
            async_tool_calls.append(self._a_execute_tool_call(tool_call))
        if async_tool_calls:
            tool_returns = await asyncio.gather(*async_tool_calls)
            return True, {
                "role": "tool",
                "tool_responses": tool_returns,
                "content": "\n\n".join([self._str_for_tool_response(tool_return) for tool_return in tool_returns]),
            }

        return False, None

    def check_termination_and_human_reply(
        self,
        messages: list[dict] | None = None,
        sender: Agent | None = None,
        config: Any | None = None,
    ) -> tuple[bool, str | None]:
        """Check if the conversation should be terminated, and if human reply is provided.

        This method checks for conditions that require the conversation to be terminated, such as reaching
        a maximum number of consecutive auto-replies or encountering a termination message. Additionally,
        it prompts for and processes human input based on the configured human input mode, which can be
        'ALWAYS', 'NEVER', or 'TERMINATE'. The method also manages the consecutive auto-reply counter
        for the conversation and prints relevant messages based on the human input received.

        Args:
            - messages (list[dict] | None): A list of message dictionaries, representing the conversation history.
            - sender (Agent | None): The agent object representing the sender of the message.
            - config (Any | None): Configuration object, defaults to the current instance if not provided.

        Returns:
            - tuple[bool, str | dict | None]: A tuple containing a boolean indicating if the conversation
            should be terminated, and a human reply which can be a string, a dictionary, or None.
        """
        iostream = IOStream.get_default()

        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender] if sender else []
        message = messages[-1]
        reply = ""
        no_human_input_msg = ""
        if self._consecutive_auto_reply_counter[sender] >= self._max_consecutive_auto_reply_dict[sender]:
            reply = "exit"
        elif self._is_termination_msg(message):
            reply = "exit"

        # print the no_human_input_msg
        if no_human_input_msg:
            iostream.print(colored(f"\n>>>>>>>> {no_human_input_msg}", "red"), flush=True)

        # stop the conversation
        if reply == "exit":
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0
            return True, None

        # send the human reply
        if reply or self._max_consecutive_auto_reply_dict[sender] == 0:
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0
            # User provided a custom response, return function and tool failures indicating user interruption
            tool_returns = []
            if message.get("function_call", False):
                tool_returns.append(
                    {
                        "role": "function",
                        "name": message["function_call"].get("name", ""),
                        "content": "USER INTERRUPTED",
                    }
                )

            if message.get("tool_calls", False):
                tool_returns.extend(
                    [
                        {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": "USER INTERRUPTED"}
                        for tool_call in message["tool_calls"]
                    ]
                )

            response = {"role": "user", "content": reply}
            if tool_returns:
                response["tool_responses"] = tool_returns

            return True, response

        # increment the consecutive_auto_reply_counter
        self._consecutive_auto_reply_counter[sender] += 1

        return False, None

    async def a_check_termination_and_human_reply(
        self,
        messages: list[dict] | None = None,
        sender: Agent | None = None,
        config: Any | None = None,
    ) -> tuple[bool, str | None]:
        """(async) Check if the conversation should be terminated, and if human reply is provided.

        This method checks for conditions that require the conversation to be terminated, such as reaching
        a maximum number of consecutive auto-replies or encountering a termination message. Additionally,
        it prompts for and processes human input based on the configured human input mode, which can be
        'ALWAYS', 'NEVER', or 'TERMINATE'. The method also manages the consecutive auto-reply counter
        for the conversation and prints relevant messages based on the human input received.

        Args:
            - messages (list[dict] | None): A list of message dictionaries, representing the conversation history.
            - sender (Agent | None): The agent object representing the sender of the message.
            - config (Any | None): Configuration object, defaults to the current instance if not provided.

        Returns:
            - tuple[bool, str | dict | None]: A tuple containing a boolean indicating if the conversation
            should be terminated, and a human reply which can be a string, a dictionary, or None.
        """
        iostream = IOStream.get_default()

        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender] if sender else []
        message = messages[-1] if messages else {}
        reply = ""
        no_human_input_msg = ""
        if self._consecutive_auto_reply_counter[sender] >= self._max_consecutive_auto_reply_dict[sender]:
            reply = "exit"
        elif self._is_termination_msg(message):
            reply = "exit"

        # print the no_human_input_msg
        if no_human_input_msg:
            iostream.print(colored(f"\n>>>>>>>> {no_human_input_msg}", "red"), flush=True)

        # stop the conversation
        if reply == "exit":
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0
            return True, None

        # send the human reply
        if reply or self._max_consecutive_auto_reply_dict[sender] == 0:
            # User provided a custom response, return function and tool results indicating user interruption
            # reset the consecutive_auto_reply_counter
            self._consecutive_auto_reply_counter[sender] = 0
            tool_returns = []
            if message.get("function_call", False):
                tool_returns.append(
                    {
                        "role": "function",
                        "name": message["function_call"].get("name", ""),
                        "content": "USER INTERRUPTED",
                    }
                )

            if message.get("tool_calls", False):
                tool_returns.extend(
                    [
                        {"role": "tool", "tool_call_id": tool_call.get("id", ""), "content": "USER INTERRUPTED"}
                        for tool_call in message["tool_calls"]
                    ]
                )

            response = {"role": "user", "content": reply}
            if tool_returns:
                response["tool_responses"] = tool_returns

            return True, response

        # increment the consecutive_auto_reply_counter
        self._consecutive_auto_reply_counter[sender] += 1

        return False, None

    def generate_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: "Agent" | None = None,
        **kwargs: Any,
    ) -> str | dict | None:
        """Reply based on the conversation history and the sender.

        Either messages or sender must be provided.
        Register a reply_func with `None` as one trigger for it to be activated when `messages` is non-empty and `sender` is `None`.
        Use registered auto reply functions to generate replies.
        By default, the following functions are checked in order:
        1. check_termination_and_human_reply
        2. generate_function_call_reply (deprecated in favor of tool_calls)
        3. generate_tool_calls_reply
        5. generate_oai_reply
        Every function returns a tuple (final, reply).
        When a function returns final=False, the next function will be checked.
        So by default, termination and human reply will be checked first.
        If not terminating and human reply is skipped, execute function or code and return the result.
        AI replies are generated only when no code execution is performed.

        Args:
            messages: a list of messages in the conversation history.
            sender: sender of an Agent instance.

        Additional keyword arguments:
            exclude (List[Callable]): a list of reply functions to be excluded.

        Returns:
            str or dict or None: reply. None if no reply is generated.
        """
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]

        # Call the hookable method that gives registered hooks a chance to process the last message.
        # Message modifications do not affect the incoming messages or self._oai_messages.
        messages = self.process_last_received_message(messages)

        # Call the hookable method that gives registered hooks a chance to process all messages.
        # Message modifications do not affect the incoming messages or self._oai_messages.
        messages = self.process_all_messages_before_reply(messages)

        for reply_func_tuple in self._reply_func_list:
            reply_func = reply_func_tuple["reply_func"]
            if "exclude" in kwargs and reply_func in kwargs["exclude"]:
                continue
            if inspect.iscoroutinefunction(reply_func):
                continue
            if self._match_trigger(reply_func_tuple["trigger"], sender):
                final, reply = reply_func(self, messages=messages, sender=sender, config=reply_func_tuple["config"])
                if final:
                    return reply
        return ""

    async def a_generate_reply(
        self,
        messages: list[dict[str, Any]] | None = None,
        sender: "Agent" | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any] | None:
        """(async) Reply based on the conversation history and the sender.

        Either messages or sender must be provided.
        Register a reply_func with `None` as one trigger for it to be activated when `messages` is non-empty and `sender` is `None`.
        Use registered auto reply functions to generate replies.
        By default, the following functions are checked in order:
        1. check_termination_and_human_reply
        2. generate_function_call_reply
        3. generate_tool_calls_reply
        5. generate_oai_reply
        Every function returns a tuple (final, reply).
        When a function returns final=False, the next function will be checked.
        So by default, termination and human reply will be checked first.
        If not terminating and human reply is skipped, execute function or code and return the result.
        AI replies are generated only when no code execution is performed.

        Args:
            messages: a list of messages in the conversation history.
            sender: sender of an Agent instance.

        Additional keyword arguments:
            exclude (List[Callable]): a list of reply functions to be excluded.

        Returns:
            str or dict or None: reply. None if no reply is generated.
        """
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            raise AssertionError(error_msg)

        if messages is None:
            messages = self._oai_messages[sender]

        # Call the hookable method that gives registered hooks a chance to process all messages.
        # Message modifications do not affect the incoming messages or self._oai_messages.
        messages = await self.a_process_all_messages_before_reply(messages)

        # Call the hookable method that gives registered hooks a chance to process the last message.
        # Message modifications do not affect the incoming messages or self._oai_messages.
        messages = await self.a_process_last_received_message(messages)

        for reply_func_tuple in self._reply_func_list:
            reply_func = reply_func_tuple["reply_func"]
            if "exclude" in kwargs and reply_func in kwargs["exclude"]:
                continue

            if self._match_trigger(reply_func_tuple["trigger"], sender):
                if inspect.iscoroutinefunction(reply_func):
                    final, reply = await reply_func(
                        self, messages=messages, sender=sender, config=reply_func_tuple["config"]
                    )
                else:
                    final, reply = reply_func(self, messages=messages, sender=sender, config=reply_func_tuple["config"])
                if final:
                    return reply
        return ""

    def _match_trigger(self, trigger: None | str | type | Agent | Callable | list, sender: Agent | None) -> bool:
        """Check if the sender matches the trigger.

        Args:
            - trigger (Union[None, str, type, Agent, Callable, List]): The condition to match against the sender.
            Can be `None`, string, type, `Agent` instance, callable, or a list of these.
            - sender (Agent): The sender object or type to be matched against the trigger.

        Returns:
            - bool: Returns `True` if the sender matches the trigger, otherwise `False`.

        Raises:
            - ValueError: If the trigger type is unsupported.
        """
        if trigger is None:
            return sender is None
        elif isinstance(trigger, str):
            if sender is None:
                raise SenderRequired()
            return trigger == sender.name
        elif isinstance(trigger, type):
            return isinstance(sender, trigger)
        elif isinstance(trigger, Agent):
            # return True if the sender is the same type (class) as the trigger
            return trigger == sender
        elif isinstance(trigger, Callable):
            rst = trigger(sender)
            assert isinstance(rst, bool), f"trigger {trigger} must return a boolean value."
            return rst
        elif isinstance(trigger, list):
            return any(self._match_trigger(t, sender) for t in trigger)
        else:
            raise ValueError(f"Unsupported trigger type: {type(trigger)}")

    def get_human_input(self, prompt: str) -> str:
        """Get human input.

        Override this method to customize the way to get human input.

        Args:
            prompt (str): prompt for the human input.

        Returns:
            str: human input.
        """
        iostream = IOStream.get_default()

        reply = iostream.input(prompt)
        self._human_input.append(reply)
        return reply

    async def a_get_human_input(self, prompt: str) -> str:
        """(Async) Get human input.

        Override this method to customize the way to get human input.

        Args:
            prompt (str): prompt for the human input.

        Returns:
            str: human input.
        """
        loop = asyncio.get_running_loop()
        reply = await loop.run_in_executor(None, functools.partial(self.get_human_input, prompt))
        return reply

    def execute_function(self, func_call: dict) -> dict[str, str]:
        """Execute a function call and return the result.

        Override this function to modify the way to execute function and tool calls.

        Args:
            func_call: a dictionary extracted from openai message at "function_call" or "tool_calls" with keys "name" and "arguments".

        """

        func_name = func_call.get("name", "")
        content = f"Error: Function {func_name} not found."

        return {
            "name": func_name,
            "role": "function",
            "content": str(content),
        }

    async def a_execute_function(self, func_call):
        """Execute an async function call and return the result.

        Override this function to modify the way async functions and tools are executed.

        Args:
            func_call: a dictionary extracted from openai message at key "function_call" or "tool_calls" with keys "name" and "arguments".

        """

        func_name = func_call.get("name", "")

        content = f"Error: Function {func_name} not found."

        return {
            "name": func_name,
            "role": "function",
            "content": str(content),
        }

    def generate_init_message(self, message: dict | str | None, **kwargs) -> str | dict:
        """Generate the initial message for the agent.
        If message is None, input() will be called to get the initial message.

        Args:
            message (str or None): the message to be processed.
            **kwargs: any additional information. It has the following reserved fields:
                "carryover": a string or a list of string to specify the carryover information to be passed to this chat. It can be a string or a list of string.
                    If provided, we will combine this carryover with the "message" content when generating the initial chat
                    message.
        Returns:
            str or dict: the processed message.
        """
        if message is None:
            message = self.get_human_input(">")

        return self._handle_carryover(message, kwargs)

    def _handle_carryover(self, message: str | dict, kwargs: dict) -> str | dict:
        if not kwargs.get("carryover"):
            return message

        if isinstance(message, str):
            return self._process_carryover(message, kwargs)

        elif isinstance(message, dict):
            if isinstance(message.get("content"), str):
                # Makes sure the original message is not mutated
                message = message.copy()
                message["content"] = self._process_carryover(message["content"], kwargs)
            elif isinstance(message.get("content"), list):
                # Makes sure the original message is not mutated
                message = message.copy()
                message["content"] = self._process_multimodal_carryover(message["content"], kwargs)
        else:
            raise InvalidCarryOverType("Carryover should be a string or a list of strings.")

        return message

    def _process_carryover(self, content: str, kwargs: dict) -> str:
        # Makes sure there's a carryover
        if not kwargs.get("carryover"):
            return content

        # if carryover is string
        if isinstance(kwargs["carryover"], str):
            content += "\nContext: \n" + kwargs["carryover"]
        elif isinstance(kwargs["carryover"], list):
            content += "\nContext: \n" + ("\n").join([_post_process_carryover_item(t) for t in kwargs["carryover"]])
        else:
            raise InvalidCarryOverType(
                "Carryover should be a string or a list of strings. Not adding carryover to the message."
            )
        return content

    def _process_multimodal_carryover(self, content: list[dict], kwargs: dict) -> list[dict]:
        """Prepends the context to a multimodal message."""
        # Makes sure there's a carryover
        if not kwargs.get("carryover"):
            return content

        return [{"type": "text", "text": self._process_carryover("", kwargs)}] + content

    async def a_generate_init_message(self, message: dict | str | None, **kwargs) -> str | dict:
        """Generate the initial message for the agent.
        If message is None, input() will be called to get the initial message.

        Args:
            Please refer to `generate_init_message` for the description of the arguments.

        Returns:
            str or dict: the processed message.
        """
        if message is None:
            message = await self.a_get_human_input(">")

        return self._handle_carryover(message, kwargs)

    def update_function_signature(self, func_sig: str | dict, is_remove: None):
        """update a function_signature in the LLM configuration for function_call.

        Args:
            func_sig (str or dict): description/name of the function to update/remove to the model. See: https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions
            is_remove: whether removing the function from llm_config with name 'func_sig'

        Deprecated as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
        See https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call
        """

        if not isinstance(self.llm_config, dict):
            error_msg = "To update a function signature, agent must have an llm_config"
            raise AssertionError(error_msg)

        if is_remove:
            if "functions" not in self.llm_config.keys():
                error_msg = "The agent config doesn't have function {name}.".format(name=func_sig)
                logger.error(error_msg)
                raise AssertionError(error_msg)
            else:
                self.llm_config["functions"] = [
                    func for func in self.llm_config["functions"] if func["name"] != func_sig
                ]
        else:
            if not isinstance(func_sig, dict):
                raise ValueError(
                    f"The function signature must be of the type dict. Received function signature type {type(func_sig)}"
                )

            self._assert_valid_name(func_sig["name"])
            if "functions" in self.llm_config.keys():
                if any(func["name"] == func_sig["name"] for func in self.llm_config["functions"]):
                    warnings.warn(f"Function '{func_sig['name']}' is being overridden.", UserWarning)

                self.llm_config["functions"] = [
                    func for func in self.llm_config["functions"] if func.get("name") != func_sig["name"]
                ] + [func_sig]
            else:
                self.llm_config["functions"] = [func_sig]

        if len(self.llm_config["functions"]) == 0:
            del self.llm_config["functions"]

        self.client = OpenAIWrapper(**self.llm_config)

    def update_tool_signature(self, tool_sig: str | dict, is_remove: None):
        """update a tool_signature in the LLM configuration for tool_call.

        Args:
            tool_sig (str or dict): description/name of the tool to update/remove to the model. See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
            is_remove: whether removing the tool from llm_config with name 'tool_sig'
        """

        if not self.llm_config:
            error_msg = "To update a tool signature, agent must have an llm_config"
            raise AssertionError(error_msg)

        if is_remove:
            if "tools" not in self.llm_config.keys():
                error_msg = "The agent config doesn't have tool {name}.".format(name=tool_sig)
                logger.error(error_msg)
                raise AssertionError(error_msg)
            else:
                self.llm_config["tools"] = [
                    tool for tool in self.llm_config["tools"] if tool["function"]["name"] != tool_sig
                ]
        else:
            if not isinstance(tool_sig, dict):
                raise ValueError(
                    f"The tool signature must be of the type dict. Received tool signature type {type(tool_sig)}"
                )
            self._assert_valid_name(tool_sig["function"]["name"])
            if "tools" in self.llm_config:
                if any(tool["function"]["name"] == tool_sig["function"]["name"] for tool in self.llm_config["tools"]):
                    warnings.warn(f"Function '{tool_sig['function']['name']}' is being overridden.", UserWarning)
                self.llm_config["tools"] = [
                    tool
                    for tool in self.llm_config["tools"]
                    if tool.get("function", {}).get("name") != tool_sig["function"]["name"]
                ] + [tool_sig]
            else:
                self.llm_config["tools"] = [tool_sig]

        if len(self.llm_config["tools"]) == 0:
            del self.llm_config["tools"]

        self.client = OpenAIWrapper(**self.llm_config)

    def _wrap_function(self, func: F) -> F:
        """Wrap the function to dump the return value to json.

        Handles both sync and async functions.

        Args:
            func: the function to be wrapped.

        Returns:
            The wrapped function.
        """

        @load_basemodels_if_needed
        @functools.wraps(func)
        def _wrapped_func(*args, **kwargs):
            retval = func(*args, **kwargs)
            return serialize_to_str(retval)

        @load_basemodels_if_needed
        @functools.wraps(func)
        async def _a_wrapped_func(*args, **kwargs):
            retval = await func(*args, **kwargs)
            return serialize_to_str(retval)

        wrapped_func = _a_wrapped_func if inspect.iscoroutinefunction(func) else _wrapped_func

        # needed for testing
        wrapped_func._origin = func

        return wrapped_func

    
    def register_model_client(self, model_client_cls: ModelClient, **kwargs):
        """Register a model client.

        Args:
            model_client_cls: A custom client class that follows the Client interface
            **kwargs: The kwargs for the custom client class to be initialized with
        """
        self.client.register_model_client(model_client_cls, **kwargs)

    def register_hook(self, hookable_method: str, hook: Callable):
        """
        Registers a hook to be called by a hookable method, in order to add a capability to the agent.
        Registered hooks are kept in lists (one per hookable method), and are called in their order of registration.

        Args:
            hookable_method: A hookable method name implemented by ConversableAgent.
            hook: A method implemented by a subclass of AgentCapability.
        """
        assert hookable_method in self.hook_lists, f"{hookable_method} is not a hookable method."
        hook_list = self.hook_lists[hookable_method]
        assert hook not in hook_list, f"{hook} is already registered as a hook."

        # async hookable checks
        expected_async = hookable_method.startswith("a_")
        hook_is_async = inspect.iscoroutinefunction(hook)
        if expected_async != hook_is_async:
            context_type = "asynchronous" if expected_async else "synchronous"
            warnings.warn(
                f"Hook '{hook.__name__}' is {'asynchronous' if hook_is_async else 'synchronous'}, "
                f"but it's being registered in a {context_type} context ('{hookable_method}'). "
                "Ensure the hook matches the expected execution context.",
                UserWarning,
            )

        hook_list.append(hook)

    def process_all_messages_before_reply(self, messages: list[dict]) -> list[dict]:
        """
        Calls any registered capability hooks to process all messages, potentially modifying the messages.
        """
        hook_list = self.hook_lists["process_all_messages_before_reply"]
        # If no hooks are registered, or if there are no messages to process, return the original message list.
        if len(hook_list) == 0 or messages is None:
            return messages

        # Call each hook (in order of registration) to process the messages.
        processed_messages = messages
        for hook in hook_list:
            if inspect.iscoroutinefunction(hook):
                continue
            processed_messages = hook(processed_messages)
        return processed_messages

    async def a_process_all_messages_before_reply(self, messages: list[dict]) -> list[dict]:
        """
        Calls any registered capability hooks to process all messages, potentially modifying the messages.
        """
        hook_list = self.hook_lists["a_process_all_messages_before_reply"]
        # If no hooks are registered, or if there are no messages to process, return the original message list.
        if len(hook_list) == 0 or messages is None:
            return messages

        # Call each hook (in order of registration) to process the messages.
        processed_messages = messages
        for hook in hook_list:
            if not inspect.iscoroutinefunction(hook):
                continue
            processed_messages = await hook(processed_messages)
        return processed_messages

    def process_last_received_message(self, messages: list[dict]) -> list[dict]:
        """
        Calls any registered capability hooks to use and potentially modify the text of the last message,
        as long as the last message is not a function call or exit command.
        """

        # If any required condition is not met, return the original message list.
        hook_list = self.hook_lists["process_last_received_message"]
        if len(hook_list) == 0:
            return messages  # No hooks registered.
        if messages is None:
            return None  # No message to process.
        if len(messages) == 0:
            return messages  # No message to process.
        last_message = messages[-1]
        if "function_call" in last_message:
            return messages  # Last message is a function call.
        if "context" in last_message:
            return messages  # Last message contains a context key.
        if "content" not in last_message:
            return messages  # Last message has no content.

        user_content = last_message["content"]
        if not isinstance(user_content, str) and not isinstance(user_content, list):
            # if the user_content is a string, it is for regular LLM
            # if the user_content is a list, it should follow the multimodal LMM format.
            return messages
        if user_content == "exit":
            return messages  # Last message is an exit command.

        # Call each hook (in order of registration) to process the user's message.
        processed_user_content = user_content
        for hook in hook_list:
            if inspect.iscoroutinefunction(hook):
                continue
            processed_user_content = hook(processed_user_content)

        if processed_user_content == user_content:
            return messages  # No hooks actually modified the user's message.

        # Replace the last user message with the expanded one.
        messages = messages.copy()
        messages[-1]["content"] = processed_user_content
        return messages

    async def a_process_last_received_message(self, messages: list[dict]) -> list[dict]:
        """
        Calls any registered capability hooks to use and potentially modify the text of the last message,
        as long as the last message is not a function call or exit command.
        """

        # If any required condition is not met, return the original message list.
        hook_list = self.hook_lists["a_process_last_received_message"]
        if len(hook_list) == 0:
            return messages  # No hooks registered.
        if messages is None:
            return None  # No message to process.
        if len(messages) == 0:
            return messages  # No message to process.
        last_message = messages[-1]
        if "function_call" in last_message:
            return messages  # Last message is a function call.
        if "context" in last_message:
            return messages  # Last message contains a context key.
        if "content" not in last_message:
            return messages  # Last message has no content.

        user_content = last_message["content"]
        if not isinstance(user_content, str) and not isinstance(user_content, list):
            # if the user_content is a string, it is for regular LLM
            # if the user_content is a list, it should follow the multimodal LMM format.
            return messages
        if user_content == "exit":
            return messages  # Last message is an exit command.

        # Call each hook (in order of registration) to process the user's message.
        processed_user_content = user_content
        for hook in hook_list:
            if not inspect.iscoroutinefunction(hook):
                continue
            processed_user_content = await hook(processed_user_content)

        if processed_user_content == user_content:
            return messages  # No hooks actually modified the user's message.

        # Replace the last user message with the expanded one.
        messages = messages.copy()
        messages[-1]["content"] = processed_user_content
        return messages

    def print_usage_summary(self, mode: str | list[str] = ["actual", "total"]) -> None:
        """Print the usage summary."""
        iostream = IOStream.get_default()

        if self.client is None:
            iostream.print(f"No cost incurred from agent '{self.name}'.")
        else:
            iostream.print(f"Agent '{self.name}':")
            self.client.print_usage_summary(mode)

    def get_actual_usage(self) -> dict[str, int] | None:
        """Get the actual usage summary."""
        if self.client is None:
            return None
        else:
            return self.client.actual_usage_summary

    def get_total_usage(self) -> dict[str, int] | None:
        """Get the total usage summary."""
        if self.client is None:
            return None
        else:
            return self.client.total_usage_summary


class AssistantAgent(ConversableAgent):
    """
    AssistantAgent is a subclass of ConversableAgent configured with a default system message.
    The default system message is designed to solve a task with LLM,
    including suggesting python code blocks and debugging.
    and `code_execution_config` is default to False.
    This agent doesn't execute code by default, and expects the user to execute the code.
    """

    DEFAULT_SYSTEM_MESSAGE = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply "TERMINATE" in the end when everything is done.
    """

    DEFAULT_DESCRIPTION = "A helpful and general-purpose AI assistant that has strong language skills, Python skills, and Linux command line skills."

    def __init__(
        self,
        name: str,
        system_message: str | None = DEFAULT_SYSTEM_MESSAGE,
        llm_config: dict | Literal[False] | None = None,
    ):
        """
        Args:
            name (str): agent name.
            system_message (str): system message for the ChatCompletion inference.
                Please override this attribute if you want to reprogram the agent.
            llm_config (dict or False or None): llm inference configuration.
                Please refer to [OpenAIWrapper.create](/docs/reference/oai/client#create)
                for available options.
        """
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
        )

        if system_message == self.DEFAULT_SYSTEM_MESSAGE:
            self.description = self.DEFAULT_DESCRIPTION
