"""Private on-disk export metadata; the public hub uses the existing Index contract."""

from typing import Literal

from pydantic import Field, model_validator

from inferatlas.schema.models import Contract, Device


class ExportManifest(Contract):
    version: str
    model: str
    device: Device
    file: str = Field(pattern=r"^[^/\\]+$")
    status: Literal["ok", "triage", "missing"]
    triage_count: int = Field(ge=0)
    exit_code: Literal[0, 2, 3]

    @model_validator(mode="after")
    def consistent_status(self) -> "ExportManifest":
        if (self.status == "ok") != (self.exit_code == 0):
            raise ValueError("only successful exports may be ok")
        if self.status != "triage" and self.triage_count:
            raise ValueError("only triage exports may have blocking items")
        if self.exit_code == 2 and self.status != "triage":
            raise ValueError("exit 2 requires triage status")
        return self
