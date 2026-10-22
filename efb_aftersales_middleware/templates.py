# coding=utf-8
"""Removed in the formatting-pipeline refactor.

Templates used to live here (customer status messages, internal ticket
cards). The middleware no longer sends template cards: the AI formats
messages; the prompts now live in the ``ai.prompts.converse`` config entry
and the per-mode strategy prompts managed via the bot's /setprompt command.
This file is kept as an empty module only because the snapshot push
channel cannot delete files; nothing imports it.
"""
