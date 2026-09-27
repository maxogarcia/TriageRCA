"""Critic package. Aeon's real Critic lives alongside base.py; the interface is re-exported here."""

from src.agents.critic.base import AlwaysAcceptCritic, Critic

__all__ = ["AlwaysAcceptCritic", "Critic"]
