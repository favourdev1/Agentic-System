from agentic_system.tools.tool_models import ToolSpec
from agentic_system.tools.builders.analysis.web import build_bank_account_api

tool = ToolSpec(
    name="bank_account_api",
    builder=build_bank_account_api,
    intent="Access and filter internal bank transaction records.",
    schema_notes="Requires optional 'params' dict (bank_name, status, search, date range).",
    groups=["analysis_plus_api"],
)
