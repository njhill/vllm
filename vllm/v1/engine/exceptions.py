# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.exceptions import VLLMServerError


class EngineGenerateError(VLLMServerError):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


class EngineDeadError(VLLMServerError):
    """Raised when the EngineCore dies. Unrecoverable."""

    message = (
        "EngineCore encountered an issue. See stack trace (above) for the root cause."
    )

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        super().__init__(self.message, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context


class EngineShutdown(EngineDeadError):
    """Raised when the EngineCore has been deliberately shut down."""

    message = "EngineCore has been shut down."
