"""CyberCodeMini License Verification & Policy Engine

Configurable engineering policy for dataset license verification and admission.
Enforces default exclusion for unknown, non-commercial, or unverified licensing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

ALLOWED_LICENSES = {
    "mit",
    "apache-2.0",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc0-1.0",
    "public-domain",
    "unlicense",
}

REQUIRE_REVIEW_LICENSES = {
    "cc-by-sa-4.0",
    "gpl-2.0",
    "gpl-3.0",
    "agpl-3.0",
    "unknown",
    "unclear",
    "none",
}

REJECT_LICENSES = {
    "cc-by-nc-4.0",
    "cc-by-nc-sa-4.0",
    "research-only",
    "non-commercial",
    "restricted",
}


@dataclass
class LicenseVerificationResult:
    """Result of dataset license policy inspection."""

    license_name: str
    status: str  # "approved" | "review" | "rejected"
    is_commercial: bool
    verified: bool
    source_notes: str
    upstream_licenses: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LicensePolicyValidator:
    """Validator enforcing engineering policy for dataset admission."""

    def __init__(self, custom_policy: Optional[dict[str, Any]] = None) -> None:
        self.allowed = ALLOWED_LICENSES
        self.require_review = REQUIRE_REVIEW_LICENSES
        self.reject = REJECT_LICENSES

        if custom_policy:
            policy = custom_policy.get("license_policy", {})
            if "allow" in policy:
                self.allowed = {l.strip().lower() for l in policy["allow"]}
            if "require_review" in policy:
                self.require_review = {l.strip().lower() for l in policy["require_review"]}
            if "reject" in policy:
                self.reject = {l.strip().lower() for l in policy["reject"]}

    def verify_license(
        self,
        dataset_license: str,
        upstream_licenses: Optional[list[str]] = None,
        license_source: str = "dataset_card",
    ) -> LicenseVerificationResult:
        """Verify license against policy rules."""
        lic_clean = dataset_license.strip().lower() if dataset_license else "unknown"
        up_list = [u.strip().lower() for u in (upstream_licenses or [])]

        is_verified = bool(lic_clean and lic_clean not in ("unknown", "unclear", "none"))

        if lic_clean in self.reject or any(u in self.reject for u in up_list):
            return LicenseVerificationResult(
                license_name=dataset_license,
                status="rejected",
                is_commercial=False,
                verified=is_verified,
                source_notes=f"Rejected by license policy (source: {license_source}).",
                upstream_licenses=up_list,
            )

        if lic_clean in self.require_review or any(u in self.require_review for u in up_list) or not is_verified:
            return LicenseVerificationResult(
                license_name=dataset_license,
                status="review",
                is_commercial=False,
                verified=is_verified,
                source_notes=f"Routed to human review (source: {license_source}).",
                upstream_licenses=up_list,
            )

        if lic_clean in self.allowed and all(u in self.allowed for u in up_list if u):
            return LicenseVerificationResult(
                license_name=dataset_license,
                status="approved",
                is_commercial=True,
                verified=True,
                source_notes=f"Approved under policy (source: {license_source}).",
                upstream_licenses=up_list,
            )

        # Default to review if unclassified
        return LicenseVerificationResult(
            license_name=dataset_license,
            status="review",
            is_commercial=False,
            verified=is_verified,
            source_notes=f"Unclassified license '{dataset_license}'; routed to review.",
            upstream_licenses=up_list,
        )
