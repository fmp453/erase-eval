import asyncio
import datetime
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from termcolor import colored

from train_methods.legacy_autogen.stream import IOStream

def consolidate_chat_info(chat_info, uniform_sender=None) -> None:
    if isinstance(chat_info, dict):
        chat_info = [chat_info]
    for c in chat_info:
        if uniform_sender is None:
            assert "sender" in c, "sender must be provided."
            sender = c["sender"]
        else:
            sender = uniform_sender
        assert "recipient" in c, "recipient must be provided."
        summary_method = c.get("summary_method")
        assert (
            summary_method is None
            or isinstance(summary_method, Callable)
            or summary_method in ("last_msg", "reflection_with_llm")
        ), "summary_method must be a string chosen from 'reflection_with_llm' or 'last_msg' or a callable, or None."
        if summary_method == "reflection_with_llm":
            assert (
                sender.client is not None or c["recipient"].client is not None
            ), "llm client must be set in either the recipient or sender when summary_method is reflection_with_llm."


Prerequisite = tuple[int, int]

@dataclass
class ChatResult:

    chat_id: int = None
    """chat id"""
    chat_history: list[dict[str, Any]] = None
    """The chat history."""
    summary: str = None
    """A summary obtained from the chat."""
    cost: dict[str, dict] = None  # keys: "usage_including_cached_inference", "usage_excluding_cached_inference"
    """The cost of the chat.
       The value for each usage type is a dictionary containing cost information for that specific type.
           - "usage_including_cached_inference": Cost information on the total usage, including the tokens in cached inference.
           - "usage_excluding_cached_inference": Cost information on the usage of tokens, excluding the tokens in cache. No larger than "usage_including_cached_inference".
    """
    human_input: list[str] = None
    """A list of human input solicited during the chat."""


def _validate_recipients(chat_queue: list[dict[str, Any]]) -> None:
    """
    Validate recipients exits and warn repetitive recipients.
    """
    receipts_set = set()
    for chat_info in chat_queue:
        assert "recipient" in chat_info, "recipient must be provided."
        receipts_set.add(chat_info["recipient"])
    if len(receipts_set) < len(chat_queue):
        warnings.warn(
            "Repetitive recipients detected: The chat history will be cleared by default if a recipient appears more than once. To retain the chat history, please set 'clear_history=False' in the configuration of the repeating agent.",
            UserWarning,
        )


def __create_async_prerequisites(chat_queue: list[dict[str, Any]]) -> list[Prerequisite]:
    """
    Create list of Prerequisite (prerequisite_chat_id, chat_id)
    """
    prerequisites = []
    for chat_info in chat_queue:
        if "chat_id" not in chat_info:
            raise ValueError("Each chat must have a unique id for async multi-chat execution.")
        chat_id = chat_info["chat_id"]
        pre_chats = chat_info.get("prerequisites", [])
        for pre_chat_id in pre_chats:
            if not isinstance(pre_chat_id, int):
                raise ValueError("Prerequisite chat id is not int.")
            prerequisites.append((chat_id, pre_chat_id))
    return prerequisites


def __find_async_chat_order(chat_ids: set[int], prerequisites: list[Prerequisite]) -> list[int]:
    """Find chat order for async execution based on the prerequisite chats

    args:
        num_chats: number of chats
        prerequisites: list of Prerequisite (prerequisite_chat_id, chat_id)

    returns:
        list: a list of chat_id in order.
    """
    edges = defaultdict(set)
    indegree = defaultdict(int)
    for pair in prerequisites:
        chat, pre = pair[0], pair[1]
        if chat not in edges[pre]:
            indegree[chat] += 1
            edges[pre].add(chat)
    bfs = [i for i in chat_ids if i not in indegree]
    chat_order = []
    steps = len(indegree)
    for _ in range(steps + 1):
        if not bfs:
            break
        chat_order.extend(bfs)
        nxt = []
        for node in bfs:
            if node in edges:
                for course in edges[node]:
                    indegree[course] -= 1
                    if indegree[course] == 0:
                        nxt.append(course)
                        indegree.pop(course)
                edges.pop(node)
        bfs = nxt

    if indegree:
        return []
    return chat_order


def _post_process_carryover_item(carryover_item):
    if isinstance(carryover_item, str):
        return carryover_item
    elif isinstance(carryover_item, dict) and "content" in carryover_item:
        return str(carryover_item["content"])
    else:
        return str(carryover_item)


def __post_carryover_processing(chat_info: dict[str, Any]) -> None:
    iostream = IOStream.get_default()

    print_carryover = (
        ("\n").join([_post_process_carryover_item(t) for t in chat_info["carryover"]])
        if isinstance(chat_info["carryover"], list)
        else chat_info["carryover"]
    )
    message = chat_info.get("message")
    if isinstance(message, str):
        print_message = message
    elif callable(message):
        print_message = "Callable: " + message.__name__
    elif isinstance(message, dict):
        print_message = "dict: " + str(message)
    elif message is None:
        print_message = "None"
    iostream.print(colored("\n" + "*" * 80, "blue"), flush=True, sep="")
    iostream.print(
        colored(
            "Starting a new chat....",
            "blue",
        ),
        flush=True,
    )
    if chat_info.get("verbose", False):
        iostream.print(colored("Message:\n" + print_message, "blue"), flush=True)
        iostream.print(colored("Carryover:\n" + print_carryover, "blue"), flush=True)
    iostream.print(colored("\n" + "*" * 80, "blue"), flush=True, sep="")


def initiate_chats(chat_queue: list[dict[str, Any]]) -> list[ChatResult]:
    """Initiate a list of chats.
    Args:
        chat_queue (list[dict]): A list of dictionaries containing the information about the chats.

        Each dictionary should contain the input arguments for
        [`ConversableAgent.initiate_chat`](/docs/reference/agentchat/conversable_agent#initiate_chat).
        For example:
            - `"sender"` - the sender agent.
            - `"recipient"` - the recipient agent.
            - `"clear_history"` (bool) - whether to clear the chat history with the agent.
               Default is True.
            - `"silent"` (bool or None) - (Experimental) whether to print the messages in this
               conversation. Default is False.
            - `"cache"` (Cache or None) - the cache client to use for this conversation.
               Default is None.
            - `"max_turns"` (int or None) - maximum number of turns for the chat. If None, the chat
               will continue until a termination condition is met. Default is None.
            - `"summary_method"` (str or callable) - a string or callable specifying the method to get
               a summary from the chat. Default is DEFAULT_summary_method, i.e., "last_msg".
            - `"summary_args"` (dict) - a dictionary of arguments to be passed to the summary_method.
               Default is {}.
            - `"message"` (str, callable or None) - if None, input() will be called to get the
               initial message.
            - `**context` - additional context information to be passed to the chat.
            - `"carryover"` - It can be used to specify the carryover information to be passed
               to this chat. If provided, we will combine this carryover with the "message" content when
               generating the initial chat message in `generate_init_message`.
            - `"finished_chat_indexes_to_exclude_from_carryover"` - It can be used by specifying a list of indexes of the finished_chats list,
               from which to exclude the summaries for carryover. If 'finished_chat_indexes_to_exclude_from_carryover' is not provided or an empty list,
               then summary from all the finished chats will be taken.
    Returns:
        (list): a list of ChatResult objects corresponding to the finished chats in the chat_queue.
    """

    consolidate_chat_info(chat_queue)
    _validate_recipients(chat_queue)
    current_chat_queue = chat_queue.copy()
    finished_chats = []
    while current_chat_queue:
        chat_info = current_chat_queue.pop(0)
        _chat_carryover = chat_info.get("carryover", [])
        finished_chat_indexes_to_exclude_from_carryover = chat_info.get(
            "finished_chat_indexes_to_exclude_from_carryover", []
        )

        if isinstance(_chat_carryover, str):
            _chat_carryover = [_chat_carryover]
        chat_info["carryover"] = _chat_carryover + [
            r.summary for i, r in enumerate(finished_chats) if i not in finished_chat_indexes_to_exclude_from_carryover
        ]
        __post_carryover_processing(chat_info)

        sender = chat_info["sender"]
        chat_res = sender.initiate_chat(**chat_info)
        finished_chats.append(chat_res)
    return finished_chats


def _on_chat_future_done(chat_future: asyncio.Future, chat_id: int):
    """
    Update ChatResult when async Task for Chat is completed.
    """
    print(f"Update chat {chat_id} result on task completion.  System time at {datetime.datetime.now()}.")
    chat_result = chat_future.result()
    chat_result.chat_id = chat_id


async def _dependent_chat_future(
    chat_id: int, chat_info: dict[str, Any], prerequisite_chat_futures: dict[int, asyncio.Future]
) -> asyncio.Task:
    """
    Create an async Task for each chat.
    """
    print(f"Create Task for chat {chat_id}.  System time at {datetime.datetime.now()}.")
    _chat_carryover = chat_info.get("carryover", [])
    finished_chat_indexes_to_exclude_from_carryover = chat_info.get(
        "finished_chat_indexes_to_exclude_from_carryover", []
    )
    finished_chats = dict()
    for chat in prerequisite_chat_futures:
        chat_future = prerequisite_chat_futures[chat]
        if chat_future.cancelled():
            raise RuntimeError(f"Chat {chat} is cancelled.")

        # wait for prerequisite chat results for the new chat carryover
        finished_chats[chat] = await chat_future

    if isinstance(_chat_carryover, str):
        _chat_carryover = [_chat_carryover]
    data = [
        chat_result.summary
        for chat_id, chat_result in finished_chats.items()
        if chat_id not in finished_chat_indexes_to_exclude_from_carryover
    ]
    chat_info["carryover"] = _chat_carryover + data
    if not chat_info.get("silent", False):
        __post_carryover_processing(chat_info)

    sender = chat_info["sender"]
    chat_res_future = asyncio.create_task(sender.a_initiate_chat(**chat_info))
    call_back_with_args = partial(_on_chat_future_done, chat_id=chat_id)
    chat_res_future.add_done_callback(call_back_with_args)
    print(f"Task for chat {chat_id} created.  System time at {datetime.datetime.now()}.")
    return chat_res_future


async def a_initiate_chats(chat_queue: list[dict[str, Any]]) -> dict[int, ChatResult]:
    consolidate_chat_info(chat_queue)
    _validate_recipients(chat_queue)
    chat_book = {chat_info["chat_id"]: chat_info for chat_info in chat_queue}
    num_chats = chat_book.keys()
    prerequisites = __create_async_prerequisites(chat_queue)
    chat_order_by_id = __find_async_chat_order(num_chats, prerequisites)
    finished_chat_futures = dict()
    for chat_id in chat_order_by_id:
        chat_info = chat_book[chat_id]
        prerequisite_chat_ids = chat_info.get("prerequisites", [])
        pre_chat_futures = dict()
        for pre_chat_id in prerequisite_chat_ids:
            pre_chat_future = finished_chat_futures[pre_chat_id]
            pre_chat_futures[pre_chat_id] = pre_chat_future
        current_chat_future = await _dependent_chat_future(chat_id, chat_info, pre_chat_futures)
        finished_chat_futures[chat_id] = current_chat_future
    await asyncio.gather(*list(finished_chat_futures.values()))
    finished_chats = dict()
    for chat in finished_chat_futures:
        chat_result = finished_chat_futures[chat].result()
        finished_chats[chat] = chat_result
    return finished_chats
