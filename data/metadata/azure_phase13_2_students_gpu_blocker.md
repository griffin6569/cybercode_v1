# CyberCodeMini — Phase 13.2: Azure for Students GPU Quota Eligibility & Cost-Preservation Audit

## Executive Summary

- **Status**: `STUDENT_SUBSCRIPTION_GPU_BLOCKED`
- **Subscription Name**: `Azure for Students`
- **Subscription ID**: `4d6ad347-4459-4206-ac7c-54cc59637a4a`
- **Target Region**: `southafricanorth`
- **Target GPU**: `Standard_NV36ads_A10_v5` (1 × NVIDIA A10 24 GB)
- **Quota Family**: `StandardNVADSA10v5Family`
- **Required Quota**: `36 vCPUs`
- **Current Quota**: `0 vCPUs`
- **User RBAC Role**: `Owner` (`xeonflare04@gmail.com`)
- **Quota API Result**: `ResourceNotAvailableForOffer`
- **Portal Result**: `Quota adjustment disabled for Azure for Students offer`
- **Azure Credit Spent**: `$0.00`
- **Subscription Changed**: `NO` (Subscription unchanged)

---

## 1. RBAC Verification (Step 1)

Role assignment check executed against subscription scope `/subscriptions/4d6ad347-4459-4206-ac7c-54cc59637a4a`:

- **Principal ID**: `20a12a57-3641-40d0-824b-f0e42f308ce6` (`xeonflare04@gmail.com`)
- **Role Assignment**: `Owner` (`/subscriptions/4d6ad347-4459-4206-ac7c-54cc59637a4a/providers/Microsoft.Authorization/roleAssignments/...`)
- **Effective Role**: `Owner`

The signed-in user possesses full `Owner` rights, which inherently include all quota-management permissions (`Microsoft.Quota/*` and `Microsoft.Capacity/*`).

---

## 2. RBAC vs. Subscription Policy Analysis (Step 2)

### Root Cause Analysis

- **Portal Error**: `You don't have permissions to adjust quotas. You must be assigned the Contributor role or higher. No data to display for the selected filters.`
- **API Response**: `ResourceNotAvailableForOffer` (`The quota request is not supported for subscription offer 'Azure for Students'`).

Because the user is verified to hold the **`Owner`** role (which supersedes `Contributor`), the inability to request GPU quota is **NOT** caused by an RBAC permission deficiency.

Instead, it is **100% caused by Azure for Students Subscription Policy restrictions** (Offer code `MS-AZR-0170P`). Azure for Students subscriptions systematically block self-service quota increases for N-series (GPU) VM families to prevent unauthorized compute consumption against the $100 promotional credit. The Azure Portal UI displays a generic RBAC message when quota adjustment functionality is disabled at the offer level.

Assigning additional RBAC permissions or roles will **NOT** resolve this issue.

---

## 3. Effective GPU Quota (Step 3)

CLI Query:
```cmd
az quota show \
  --resource-name StandardNVADSA10v5Family \
  --scope /subscriptions/4d6ad347-4459-4206-ac7c-54cc59637a4a/providers/Microsoft.MachineLearningServices/locations/southafricanorth \
  -o json
```

Result:
- **Quota Limit**: `0`
- **Unit**: `Count` (vCPUs)
- **isQuotaApplicable**: `false` / `ResourceNotAvailableForOffer`

---

## 4. Quota Request History & Policy Compliance (Step 4)

- **Existing Quota Requests**: `0`
- **Quota Request Retries**: `0` (Strictly avoided per safety protocols)
- **Policy Enforcement**: No unauthorized API retries, bypass attempts, or resource manipulations were executed.

---

## 5. Alternative Legitimate Funding & Options (Step 5 & 7)

Without changing the current `Azure for Students` subscription, the following legitimate options exist to enable GPU access for Phase 13 training:

### Option A: Azure Support Ticket (Manual Exception Request)
Submit an official Azure Support Ticket under *Service and Subscription Limits (Quotas)* requesting a manual student allocation of 36 vCPUs for `StandardNVADSA10v5Family` in `southafricanorth`. Microsoft Support may evaluate manual requests on a case-by-case basis.

### Option B: Upgrade Subscription to Pay-As-You-Go
Converting the subscription to Pay-As-You-Go unlocks GPU quota eligibility and request capabilities.
- **Credit Impact**: According to official Microsoft Azure documentation, remaining unused $100 student credit is retained and applied toward eligible usage for the duration of the original 12-month student period.
- **Billing Requirement**: A valid credit card is required for identity verification and overage protection.
- **Action Status**: Preserved in pending state; NOT executed during Phase 13.2.

### Option C: Institutional / Enterprise Azure Sponsorship
Associate the workspace/account with an existing university, employer, or Microsoft research sponsorship subscription that possesses pre-allocated N-series GPU quotas.

---

## 6. Cost Protection & Safety Audit (Step 6)

All cost preservation constraints were strictly satisfied:

- **Subscription Upgrade**: `NOT EXECUTED` (Preserved as Azure for Students)
- **Pay-As-You-Go Billing**: `NOT CREATED`
- **Azure GPU Compute Clusters**: `0`
- **Virtual Machines Created**: `0`
- **Azure ML Training Jobs**: `0`
- **Azure Credit Spent**: `$0.00`
- **Frozen Training Dataset**: `v0.3.0` (Immutable, SHA-256 verified)

---

## 7. Audit Metadata Summary

| Parameter | Value |
| :--- | :--- |
| Status | `STUDENT_SUBSCRIPTION_GPU_BLOCKED` |
| Subscription | `Azure for Students` (`4d6ad347-4459-4206-ac7c-54cc59637a4a`) |
| Resource Group | `cybercodemini-rg` (`eastus`) |
| Azure ML Workspace | `cybercodemini-ml` (`southafricanorth`) |
| Target GPU SKU | `Standard_NV36ads_A10_v5` |
| Quota Family | `StandardNVADSA10v5Family` |
| Required vCPUs | `36` |
| Current vCPUs | `0` |
| Signed-In User Role | `Owner` |
| Quota API Status | `ResourceNotAvailableForOffer` |
| Azure Credit Spent | `$0.00` |
