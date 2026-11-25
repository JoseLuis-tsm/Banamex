"""Evidencia determinista para el asistente (contrato: docs/contratos/evidencia.md)."""

from src.evidence.contract import CONSULTAS, ESTADOS, validate_evidence
from src.evidence.service import get_evidence, list_mock_names

__all__ = ["CONSULTAS", "ESTADOS", "get_evidence", "list_mock_names", "validate_evidence"]
