"""Private on-disk export metadata; the public hub uses the existing Index contract."""

from typing import Literal

from pydantic import Field, model_validator

from inferatlas.schema.models import Contract, Device

EXPORT_FILENAME = r"^[A-Za-z0-9_][A-Za-z0-9_.-]*\.html$"


class ExportManifest(Contract):
    version: str
    model: str
    device: Device
    file: str = Field(pattern=EXPORT_FILENAME)
    status: Literal["ok", "triage", "missing"]
    triage_count: int = Field(ge=0)
    exit_code: Literal[0, 2, 3]

    @model_validator(mode="after")
    def consistent_status(self) -> "ExportManifest":
        if ".." in self.file:
            raise ValueError("export filename must not contain '..' (the web hub rejects it)")
        if (self.status == "ok" and self.exit_code != 0) or (
            self.status == "missing" and self.exit_code == 0
        ):
            raise ValueError("successful exports require ok or triage status")
        if self.status != "triage" and self.triage_count:
            raise ValueError("only triage exports may have review notes")
        if self.exit_code == 2 and self.status != "triage":
            raise ValueError("exit 2 requires triage status")
        return self
